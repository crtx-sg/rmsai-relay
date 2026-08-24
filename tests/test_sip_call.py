"""On-demand SIP call (phone pipeline, Phase A) — request shape, room naming, and composition.

The dial itself needs a carrier trunk, so what is pinned here is everything that decides whether
that dial can succeed: the fields handed to `CreateSIPParticipantRequest` (a real trunk rejects a
call with no caller ID, and an unbounded call bills until someone notices), the room the agent is
dispatched into, and the ordering that keeps the agent from waiting in the wrong room.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from common.audit import AuditLog
from common.config import DEFAULT
from voice.livekit_cloud import build_sip_participant_kwargs
from voice.outbound import (
    CallOutcome,
    SimulatedCaller,
    call_room_name,
    mask_number,
    place_predefined_call,
)

_CFG = replace(DEFAULT, livekit_sip_trunk_id="ST_abc123", outbound_from="+15550001111",
               outbound_call_number="+15559998888", sip_ringing_timeout_s=25,
               sip_max_call_duration_s=300)


# --- the SIP request ----------------------------------------------------------------------------

def test_request_carries_trunk_destination_and_room():
    kw = build_sip_participant_kwargs(room="rmsai-call-abc", number="+15559998888", config=_CFG)
    assert kw["sip_trunk_id"] == "ST_abc123"
    assert kw["sip_call_to"] == "+15559998888"
    assert kw["room_name"] == "rmsai-call-abc"
    assert kw["wait_until_answered"] is True


def test_caller_id_is_sent_when_configured():
    # Most carrier trunks drop a call presenting no valid from-number — this is the field that was
    # missing while OUTBOUND_FROM sat configured but unused, so the trunk looked broken.
    assert build_sip_participant_kwargs(room="r", number="+1555", config=_CFG)["sip_number"] == \
        "+15550001111"
    # ...and omitted entirely (not sent empty) when unset, so the trunk's own default applies.
    assert "sip_number" not in build_sip_participant_kwargs(
        room="r", number="+1555", config=replace(_CFG, outbound_from=""))


def test_call_is_time_bounded_at_both_ends():
    kw = build_sip_participant_kwargs(room="r", number="+1555", config=_CFG)
    assert kw["ringing_timeout"].seconds == 25      # unanswered call releases the trunk channel
    assert kw["max_call_duration"].seconds == 300   # backstop against a voicemail billed forever


def test_zero_timeout_leaves_the_field_unset():
    kw = build_sip_participant_kwargs(
        room="r", number="+1555", config=replace(_CFG, sip_max_call_duration_s=0))
    assert kw["max_call_duration"] is None


def test_missing_trunk_fails_fast():
    # Fail here rather than let the SDK report a generic dial error that the retry policy would
    # then dutifully repeat three times against a trunk that does not exist.
    with pytest.raises(ValueError, match="LIVEKIT_SIP_TRUNK_ID"):
        build_sip_participant_kwargs(room="r", number="+1555",
                                     config=replace(_CFG, livekit_sip_trunk_id=""))


# --- room + composition -------------------------------------------------------------------------

def test_room_is_one_per_call_under_the_shared_prefix():
    assert call_room_name("abc123", _CFG) == "rmsai-call-abc123"
    assert call_room_name("abc123", replace(_CFG, call_room_prefix="x-")) == "x-abc123"


def test_agent_is_dispatched_into_the_room_that_is_dialled(tmp_path):
    # The agent must be in the room BEFORE the callee answers, and in *this* room — a mismatch
    # leaves it waiting somewhere the call never enters, which sounds exactly like a dead line.
    dispatched: list[str] = []
    room, outcome, attempts = place_predefined_call(
        config=_CFG, caller=SimulatedCaller(), dispatcher=dispatched.append,
        audit=AuditLog(str(tmp_path / "audit.jsonl")), sleep_fn=lambda _s: None,
    )
    assert dispatched == [room] and room.startswith("rmsai-call-")
    assert outcome == CallOutcome.ANSWERED and attempts == 1


def test_dispatch_failure_does_not_swallow_the_call(tmp_path):
    def _boom(_room):
        raise RuntimeError("livekit unreachable")

    _room, outcome, _attempts = place_predefined_call(
        config=_CFG, caller=SimulatedCaller(), dispatcher=_boom,
        audit=AuditLog(str(tmp_path / "audit.jsonl")), sleep_fn=lambda _s: None,
    )
    assert outcome == CallOutcome.ANSWERED  # the phone still rings; only the greeting is at risk


def test_no_answer_retries_then_audits_the_masked_destination(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    caller = SimulatedCaller([CallOutcome.NO_ANSWER, CallOutcome.NO_ANSWER, CallOutcome.NO_ANSWER])
    _room, outcome, attempts = place_predefined_call(
        config=_CFG, caller=caller, audit=audit, sleep_fn=lambda _s: None,
    )
    assert outcome == CallOutcome.NO_ANSWER and attempts == 3  # 1 + OUTBOUND_MAX_RETRIES
    line = audit.read_all()[-1]
    assert line["action"] == "outbound_call" and line["outcome"] == "no_answer"
    assert line["extra"]["to"] == "+1555…8888"          # masked: enough to identify, not to dial
    assert "9998" not in line["extra"]["to"]
    assert line["extra"]["room"].startswith("rmsai-call-") and line["extra"]["attempts"] == 3


def test_call_without_a_destination_is_refused(tmp_path):
    with pytest.raises(ValueError, match="OUTBOUND_CALL_NUMBER"):
        place_predefined_call(config=replace(_CFG, outbound_call_number=""),
                              caller=SimulatedCaller(),
                              audit=AuditLog(str(tmp_path / "audit.jsonl")))


def test_mask_keeps_enough_to_tell_destinations_apart():
    assert mask_number("+15559998888") == "+1555…8888"
    assert mask_number("") == "…"
