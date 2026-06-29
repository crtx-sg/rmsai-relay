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
from common.criticality import at_least, event_criticality, vitals_override
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
    dropped: bool = False  # call dropped mid-alert
    status: str = "reported"
    transcript: list[str] = field(default_factory=list)  # what the agent said/sent


def should_call(event: DeviceEvent, config: Config = DEFAULT) -> tuple[bool, str]:
    """Decide whether this event dials out. Returns (call?, reason).

    A false-positive ECG (NORMAL_SINUS) normally does not call (spec D10). But when
    `criticality_fp_override_on_vitals` is on, a vitals-driven escalation (MEWS >= threshold or a
    deteriorating trend) **overrides** that guard — the patient is deteriorating regardless of the
    rhythm, so we still call. The returned reason names the override for the audit/console log.
    """
    if not config.outbound_enabled:
        return False, "outbound_disabled"
    crit = event_criticality(event, config)
    vitals_warn, why = vitals_override(event, config)
    fp_override = event.is_false_positive and config.criticality_fp_override_on_vitals and vitals_warn
    if event.is_false_positive and not fp_override:
        return False, "false_positive"
    if not at_least(crit, config.outbound_min_criticality):
        return False, f"below_threshold ({crit} < {config.outbound_min_criticality})"
    return True, (f"fp_override ({why})" if fp_override else "ok")


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
    drop_after: int | None = None,
    live_audio: bool = False,
) -> OutboundResult:
    """Run the outbound loop. `utterances` is the clinician's scripted side of the call.

    `drop_after` simulates the call dropping after that many clinician turns (mid-alert): the
    event stays `reported` (unacknowledged) and is eligible for the same retry policy.

    `live_audio=True` is the real-LiveKit path: the relay only *places* the call; the agent worker
    that joins the room drives the actual PIN -> alert -> Q&A -> ack audio loop (and writes the ack
    status from its side). So once the call is answered there is no scripted `_converse` here — the
    event is left `reported` and the worker promotes it to `acknowledged`.
    """
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

    if live_audio:
        # Worker owns the conversation from here (real STT/TTS over LiveKit). Don't script it.
        audit.write(actor="system", action="outbound_live", subject=event.window.patient_ref,
                    outcome="worker_driven")
        alert = spoken_report(event, bed=bed, config=config)
        return OutboundResult(called=True, decision_reason=reason, outcome=outcome.value,
                              attempts=attempts, status="reported", spoken_report=alert,
                              transcript=[alert])

    conv = _converse(event, orchestrator, utterances, session_id, bed,
                     emit=lambda _t: None, drop_after=drop_after, config=config)
    set_event_status(driver, uuid, conv["status"])
    ack_outcome = "dropped" if conv["dropped"] else conv["status"]
    audit.write(actor="system", action="acknowledgment", subject=event.window.patient_ref,
                outcome=ack_outcome)
    return OutboundResult(
        called=True, decision_reason=reason, outcome=outcome.value, attempts=attempts,
        spoken_report=conv["transcript"][0], answers=conv["answers"],
        acknowledged=conv["acknowledged"], dropped=conv["dropped"],
        status=conv["status"], transcript=conv["transcript"],
    )


def _converse(event, orchestrator, utterances, session_id, bed, *, emit, drop_after=None,
              config: Config = DEFAULT) -> dict:
    """Shared report -> follow-ups -> ack loop. `emit(text)` delivers each agent message.

    Returns {answers, acknowledged, status, transcript}. Channel-agnostic: voice records the
    transcript (emit is a no-op), text sends each message as an SMS.
    """
    transcript: list[str] = []

    def say(text: str) -> None:
        transcript.append(text)
        emit(text)

    say(spoken_report(event, bed=bed, config=config))
    answers: list[str] = []
    acknowledged = False
    dropped = False
    status = "reported"
    idx = 0

    def _next() -> str | None:
        nonlocal idx
        if idx >= len(utterances) or idx >= _MAX_TURNS:
            return None
        u = utterances[idx]
        idx += 1
        return u

    while True:
        if drop_after is not None and idx >= drop_after:
            dropped = True  # caller hung up mid-alert -> stays reported (unacknowledged)
            break
        u = _next()
        if u is None:
            break
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

    return {"answers": answers, "acknowledged": acknowledged, "dropped": dropped,
            "status": status, "transcript": transcript}


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

    alert = spoken_report(event, bed=bed, config=config)
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

    conv = _converse(event, orchestrator, utterances, session_id, bed, emit=emit, config=config)
    set_event_status(driver, uuid, conv["status"])
    audit.write(actor="system", action="acknowledgment", subject=event.window.patient_ref,
                outcome=conv["status"])
    return OutboundResult(
        called=True, decision_reason=reason, channel="text", outcome="delivered",
        spoken_report=alert, answers=conv["answers"], acknowledged=conv["acknowledged"],
        status=conv["status"], transcript=conv["transcript"],
    )
