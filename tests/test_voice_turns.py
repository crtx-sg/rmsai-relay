"""Voice-turn behaviour from the live call of 2026-10-09 08:05 (worklist questions on the phone).

- the greeting gate missed one answered call (sip.callStatus never read `active`), so it now also
  opens on the caller's speech and logs what it saw;
- an 8-record worklist read aloud ran ~30s; voice turns now get a short summary;
- "Yeah"/"Okay." spoken over an answer each got "I don't have information on that…";
- "alarms in bed three" came back without the patient or bed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest
from livekit import rtc

from common.audit import AuditLog
from kb.graph.templates import TEMPLATES
from orchestrator.orchestrator import _row_to_sentence, _spoken_summary
from voice.auth import PinAuthGate
from voice.handlers import OrchestratorHandler, OutboundHandler, is_filler
from voice.livekit_agent import describe_callers, wait_for_caller

_K = rtc.ParticipantKind

# --- 1. greeting gate -----------------------------------------------------------------------------


def _ringing_room():
    return NS(remote_participants={"sip": NS(kind=_K.PARTICIPANT_KIND_SIP,
                                             attributes={"sip.callStatus": "ringing"})})


def test_gate_opens_when_the_caller_is_heard_even_if_status_lags(capsys):
    room, heard, t = _ringing_room(), asyncio.Event(), [0.0]

    async def _sleep(s):
        t[0] += s
        if t[0] >= 2.0:
            heard.set()  # the caller starts talking (e.g. says the PIN)

    assert asyncio.run(wait_for_caller(room, 70.0, heard=heard, clock=lambda: t[0], sleep=_sleep))
    assert t[0] < 3.0
    assert "SIP(sip.callStatus=ringing)" in capsys.readouterr().out


def test_gate_timeout_logs_what_it_saw(capsys):
    t = [0.0]

    async def _tick(s):
        t[0] += s

    assert not asyncio.run(wait_for_caller(_ringing_room(), 1.0, heard=asyncio.Event(),
                                           clock=lambda: t[0], sleep=_tick))
    assert "gate timed out; participants: SIP(sip.callStatus=ringing)" in capsys.readouterr().out


def test_describe_callers_has_no_identities():
    out = describe_callers([NS(kind=_K.PARTICIPANT_KIND_SIP, identity="sip_+919800000016",
                               attributes={"sip.callStatus": "active", "sip.phoneNumber": "+9198"})])
    assert out == "SIP(sip.callStatus=active)"


# --- 2. spoken summaries --------------------------------------------------------------------------

_WORKLIST = [  # as the template returns them: Critical first, newest first
    {"patient": "PT992591", "bed": "Unit1-Bed05", "event": "VENTRICULAR_TACHYCARDIA", "criticality": "Critical", "ts": 5},
    {"patient": "PT929178", "bed": "Unit1-Bed04", "event": "VENTRICULAR_FIBRILLATION", "criticality": "Critical", "ts": 4},
    {"patient": "PT998224", "bed": "Unit1-Bed02", "event": "VENTRICULAR_TACHYCARDIA", "criticality": "Critical", "ts": 3},
    {"patient": "PT942338", "bed": "Unit1-Bed01", "event": "ATRIAL_FIBRILLATION", "criticality": "High", "ts": 9},
    {"patient": "PT942338", "bed": "Unit1-Bed01", "event": "ATRIAL_FIBRILLATION", "criticality": "High", "ts": 8},
    {"patient": "PT937071", "bed": "Unit1-Bed07", "event": "SVT", "criticality": "High", "ts": 7},
]


def test_worklist_is_summarised_for_voice():
    out = _spoken_summary(_WORKLIST, "worklist")
    assert out == ("6 unacknowledged events. 3 Critical: bed 5 ventricular tachycardia, bed 4 "
                   "ventricular fibrillation, bed 2 ventricular tachycardia. 3 High: bed 1 atrial "
                   "fibrillation twice, bed 7 SVT. Ask about a bed for details.")
    assert "UTC" not in out  # no timestamps read aloud


def test_alarm_counts_name_the_leaders():
    rows = [{"patient": "PT992591", "bed": "Unit1-Bed05", "alarms": 2},
            {"patient": "PT942338", "bed": "Unit1-Bed01", "alarms": 2},
            {"patient": "PT929178", "bed": "Unit1-Bed04", "alarms": 1},
            {"patient": "PT986685", "bed": "Unit1-Bed06", "alarms": 1}]
    assert _spoken_summary(rows, "alarm_counts") == (
        "Most alarms: 2 each, on bed 5, patient PT992591 and bed 1, patient PT942338. "
        "2 other beds have fewer.")


def test_other_lists_are_cut_to_three():
    rows = [{"patient": f"PT{i}", "event": "PVC", "ts": None} for i in range(5)]
    out = _spoken_summary(rows, "events_for_patient")
    assert out.startswith("5 records.") and "And 2 more; ask for details." in out
    assert out.count("PVC") == 3


# --- 3. fillers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Yeah", "Okay.", "ok", "mm-hmm", "Uh huh", "right", "okay, yeah"])
def test_backchannel_is_filler(text):
    assert is_filler(text)


@pytest.mark.parametrize("text", ["yes", "no", "Thank you", "okay what about bed three",
                                  "1234", "Which bed has the most severe alarm?"])
def test_real_turns_are_not_filler(text):
    assert not is_filler(text)


class _Orch:
    def __init__(self):
        self.calls = []

    def handle_turn(self, session_id, text, *, spoken=False, **_kw):
        self.calls.append((text, spoken))
        return NS(answer=f"answer to {text}", declined=False)


class _Working:
    def __init__(self, authenticated):
        self.state = NS(authenticated=authenticated, patient_ref=None)

    def get_or_create(self, _sid):
        return self.state

    def set_authenticated(self, _sid, patient_ref=None):
        self.state.authenticated, self.state.patient_ref = True, patient_ref


def _handler(tmp_path, authenticated=True, cls=OrchestratorHandler, **kw):
    orch = _Orch()
    h = cls(orch, _Working(authenticated), auth_gate=PinAuthGate(NS(inbound_auth_pin="1234")),
            audit=AuditLog(tmp_path / "a.jsonl"), **kw)
    return h, orch


def test_spoken_filler_gets_no_reply_and_no_query(tmp_path):
    h, orch = _handler(tmp_path)
    assert h.respond("Yeah", session_id="s", spoken=True) == ""
    assert orch.calls == [] and AuditLog(tmp_path / "a.jsonl").read_all() == []
    # typed "ok" in text chat still goes to the orchestrator
    assert h.respond("ok", session_id="s") == "answer to ok"


def test_spoken_question_is_marked_spoken(tmp_path):
    h, orch = _handler(tmp_path)
    h.respond("what is in my worklist", session_id="s", spoken=True)
    assert orch.calls == [("what is in my worklist", True)]


def test_filler_before_the_pin_is_not_an_attempt(tmp_path):
    h, _ = _handler(tmp_path, authenticated=False)
    assert h.respond("okay", session_id="s", spoken=True) == ""
    assert h._attempts == {}


def test_outbound_yes_still_acknowledges(tmp_path):
    alert = NS(patient_ref="PT992591", event_id="e1", spoken_alert="alert")
    h, orch = _handler(tmp_path, cls=OutboundHandler, alert=alert)
    out = h.respond("yes", session_id="s", spoken=True)
    assert "recorded your acknowledgment" in out and orch.calls == []


# --- 4. bed events carry patient + bed ------------------------------------------------------------

def test_bed_events_return_patient_and_bed():
    cypher = TEMPLATES["event_status_on_bed"]
    assert "p.pseudonym AS patient" in cypher and "b.label AS bed" in cypher
    sentence = _row_to_sentence({"patient": "PT984532", "bed": "Unit1-Bed03", "ts": 1737,
                                 "reported_event": "PVC", "status": "reported"})
    assert "patient PT984532 on bed Unit1-Bed03" in sentence


# --- barge-in: "ok" stops the agent and it listens ------------------------------------------------

from voice.livekit_agent import BARGE_IN, is_barge_in  # noqa: E402


@pytest.mark.parametrize("text", ["ok", "Okay.", "OK!", "stop", "wait", "hold on"])
def test_barge_in_words(text):
    assert is_barge_in(text)
    assert is_filler(text)  # ... and they get no spoken reply, so the agent just listens


@pytest.mark.parametrize("text", ["okay what about bed three", "yes", "stop the alarm on bed 3"])
def test_sentences_are_not_barge_in(text):
    assert not is_barge_in(text)


def test_barge_in_policy_survives_slow_stt():
    # "ok" is short, and its transcript lands ~1.7-2.6s after speech: the SDK defaults (0.5s minimum,
    # 2.0s before resuming a "false" interruption) let the answer run on over it.
    assert BARGE_IN["mode"] == "vad" and BARGE_IN["min_duration"] <= 0.3
    assert BARGE_IN["false_interruption_timeout"] >= 3.0


# --- PIN with a leading interjection (live: "Oh, one, two, three, four" -> rejected) --------------

from voice.auth import parse_pin  # noqa: E402


@pytest.mark.parametrize("spoken, ok", [
    ("Oh, one, two, three, four", True), ("oh one two three four", True), ("Um, 1234", True),
    ("one two three four", True), ("Oh, one, two, three, five", False), ("oh", False),
    ("Oh, one, two, three, four, five", False),
])
def test_pin_ignores_a_leading_interjection(spoken, ok):
    assert PinAuthGate(NS(inbound_auth_pin="1234")).verify(spoken) is ok


def test_mid_pin_oh_is_still_zero():
    assert parse_pin("one oh two three") == "1023"
    assert PinAuthGate(NS(inbound_auth_pin="1023")).verify("one oh two three")


# --- call_still_up ---------------------------------------------------------------------------------

def test_call_is_up_while_a_sip_participant_remains():
    from livekit.protocol.models import ParticipantInfo

    from voice.livekit_cloud import call_still_up

    K = ParticipantInfo.Kind
    assert call_still_up([K.AGENT, K.SIP]) and not call_still_up([K.AGENT]) and not call_still_up([])


# --- every call authenticates from scratch (live 2026-10-09 09:14) ---------------------------------

class _StoredWorking(_Working):
    """Session state that outlives a call, like the Redis-backed WorkingMemory (no TTL)."""

    def clear(self, _sid):
        self.state = NS(authenticated=False, patient_ref=None)


def test_new_call_on_a_reused_room_requires_the_pin(tmp_path):
    # The previous call on rmsai-outbound-<event> left the session authenticated. Re-ingesting the
    # same event reused the room, and "one two three four" went to the KB instead of the PIN check.
    orch = _Orch()
    working = _StoredWorking(authenticated=True)
    h = OrchestratorHandler(orch, working, auth_gate=PinAuthGate(NS(inbound_auth_pin="1234")),
                            audit=AuditLog(tmp_path / "a.jsonl"))
    h.begin_session("rmsai-outbound-91f65594")
    out = h.respond("One, two, three, four", session_id="rmsai-outbound-91f65594", spoken=True)
    assert "authenticated" in out and orch.calls == []  # the PIN path, not the KB
    assert working.state.authenticated


def test_without_the_pin_a_reused_room_shares_nothing(tmp_path):
    orch = _Orch()
    h = OrchestratorHandler(orch, _StoredWorking(authenticated=True),
                            auth_gate=PinAuthGate(NS(inbound_auth_pin="1234")),
                            audit=AuditLog(tmp_path / "a.jsonl"))
    h.begin_session("room")
    out = h.respond("what is the latest alarm", session_id="room", spoken=True)
    assert "authenticate" in out and orch.calls == []
