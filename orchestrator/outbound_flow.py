"""Outbound full loop (§6.1): event -> decide -> call -> spoken report -> follow-ups -> ack.

A non-false-positive event at/above the criticality gate triggers one outbound call to the single
configured number. On answer, the agent speaks the report, takes grounded follow-up questions
(KB + memory, via the orchestrator), and records a verbal acknowledgment with one confirm-back,
setting `MonitoredEvent.status` to acknowledged / reported / notify_failed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from common.audit import AuditLog
from common.config import DEFAULT, Config
from common.criticality import at_least, criticality
from common.schemas import DeviceEvent
from kb.graph.driver import GraphDriver
from kb.graph.events import set_event_status
from voice.outbound import Caller, CallOutcome, parse_ack, place_with_retries

from .report import spoken_report

_CONFIRM_BACK = "I heard you acknowledge the alert — is that correct?"
_MAX_TURNS = 12


@dataclass
class OutboundResult:
    called: bool
    decision_reason: str
    channel: str = "voice"  # voice | text
    outcome: str | None = None
    attempts: int = 0
    spoken_report: str = ""
    answers: list[str] = field(default_factory=list)
    acknowledged: bool = False
    status: str = "reported"
    transcript: list[str] = field(default_factory=list)  # what the agent said/sent


def should_call(event: DeviceEvent, config: Config = DEFAULT) -> tuple[bool, str]:
    """Decide whether this event dials out. Returns (call?, reason)."""
    if event.is_false_positive:
        return False, "false_positive"
    if not config.outbound_enabled:
        return False, "outbound_disabled"
    crit = criticality(event.event_type, event.analysis.mews.risk)
    if not at_least(crit, config.outbound_min_criticality):
        return False, f"below_threshold ({crit} < {config.outbound_min_criticality})"
    return True, "ok"


def run_outbound(
    event: DeviceEvent,
    *,
    driver: GraphDriver,
    orchestrator,
    caller: Caller,
    utterances: list[str],
    config: Config = DEFAULT,
    bed: str | None = None,
    session_id: str | None = None,
    audit: AuditLog | None = None,
    sleep_fn=time.sleep,
) -> OutboundResult:
    """Run the outbound loop. `utterances` is the clinician's scripted side of the call."""
    audit = audit or AuditLog()
    session_id = session_id or f"outbound-{event.window.event_id}"
    uuid = event.window.event_id

    call, reason = should_call(event, config)
    if not call:
        return OutboundResult(called=False, decision_reason=reason, status="reported")

    outcome, attempts = place_with_retries(
        caller, config.outbound_call_number, config, sleep_fn=sleep_fn
    )
    audit.write(actor="system", action="outbound_call", subject=event.window.patient_ref,
                outcome=outcome.value, attempts=attempts)

    if outcome != CallOutcome.ANSWERED:
        set_event_status(driver, uuid, "notify_failed")
        return OutboundResult(called=True, decision_reason=reason, outcome=outcome.value,
                              attempts=attempts, status="notify_failed")

    conv = _converse(event, orchestrator, utterances, session_id, bed, emit=lambda _t: None)
    set_event_status(driver, uuid, conv["status"])
    audit.write(actor="system", action="acknowledgment", subject=event.window.patient_ref,
                outcome=conv["status"])
    return OutboundResult(
        called=True, decision_reason=reason, outcome=outcome.value, attempts=attempts,
        spoken_report=conv["transcript"][0], answers=conv["answers"],
        acknowledged=conv["acknowledged"], status=conv["status"], transcript=conv["transcript"],
    )


def _converse(event, orchestrator, utterances, session_id, bed, *, emit) -> dict:
    """Shared report -> follow-ups -> ack loop. `emit(text)` delivers each agent message.

    Returns {answers, acknowledged, status, transcript}. Channel-agnostic: voice records the
    transcript (emit is a no-op), text sends each message as an SMS.
    """
    transcript: list[str] = []

    def say(text: str) -> None:
        transcript.append(text)
        emit(text)

    say(spoken_report(event, bed=bed))
    answers: list[str] = []
    acknowledged = False
    status = "reported"
    idx = 0

    def _next() -> str | None:
        nonlocal idx
        if idx >= len(utterances) or idx >= _MAX_TURNS:
            return None
        u = utterances[idx]
        idx += 1
        return u

    while (u := _next()) is not None:
        intent = parse_ack(u)
        if intent == "yes":
            say(_CONFIRM_BACK)
            confirm = _next()
            if confirm is not None and parse_ack(confirm) == "yes":
                acknowledged, status = True, "acknowledged"
                break
            say(_CONFIRM_BACK)  # re-prompt once
            confirm2 = _next()
            if confirm2 is not None and parse_ack(confirm2) == "yes":
                acknowledged, status = True, "acknowledged"
            break  # leave 'reported' (unacknowledged) if still unclear
        if intent == "no":
            break
        result = orchestrator.handle_turn(session_id, u)  # grounded follow-up
        answers.append(result.answer)
        say(result.answer)

    return {"answers": answers, "acknowledged": acknowledged, "status": status,
            "transcript": transcript}


def run_text_notify(
    event: DeviceEvent,
    *,
    driver: GraphDriver,
    orchestrator,
    notifier,
    to: str,
    utterances: list[str],
    config: Config = DEFAULT,
    bed: str | None = None,
    session_id: str | None = None,
    audit: AuditLog | None = None,
) -> OutboundResult:
    """Alert by text message instead of a voice call (POC option). Same gate + ack semantics."""
    audit = audit or AuditLog()
    session_id = session_id or f"text-{event.window.event_id}"
    uuid = event.window.event_id

    call, reason = should_call(event, config)
    if not call:
        return OutboundResult(called=False, decision_reason=reason, channel="text", status="reported")

    alert = spoken_report(event, bed=bed)
    delivered = notifier.send(to, alert)
    audit.write(actor="system", action="text_notify", subject=event.window.patient_ref,
                outcome="delivered" if delivered else "failed")
    if not delivered:
        set_event_status(driver, uuid, "notify_failed")
        return OutboundResult(called=True, decision_reason=reason, channel="text",
                              outcome="failed", spoken_report=alert, status="notify_failed")

    # The report was already sent above; the loop sends the confirm-back + answers as replies.
    sent_first = {"done": False}

    def emit(text: str) -> None:
        if not sent_first["done"]:  # skip re-sending the alert (already delivered)
            sent_first["done"] = True
            return
        notifier.send(to, text)

    conv = _converse(event, orchestrator, utterances, session_id, bed, emit=emit)
    set_event_status(driver, uuid, conv["status"])
    audit.write(actor="system", action="acknowledgment", subject=event.window.patient_ref,
                outcome=conv["status"])
    return OutboundResult(
        called=True, decision_reason=reason, channel="text", outcome="delivered",
        spoken_report=alert, answers=conv["answers"], acknowledged=conv["acknowledged"],
        status=conv["status"], transcript=conv["transcript"],
    )
