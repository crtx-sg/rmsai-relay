"""Outbound LiveKit voice path (offline): alert hand-off, PIN-gated outbound handler, live run.

Covers the relay->worker seam without LiveKit/redis: the alert store round-trips, the
`OutboundHandler` voices the event after the PIN gate and records the ack, `resolve_handler`
picks outbound vs inbound by alert presence, and the consumer's live path stages the alert +
places the call without scripting the conversation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from common.audit import AuditLog
from common.config import DEFAULT
from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.serialize import event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator import bus_consumer
from voice.outbound import CallOutcome, SimulatedCaller
from voice.outbound_alert import OutboundAlert, OutboundAlertStore

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_CFG = replace(DEFAULT, outbound_enabled=True, outbound_call_number="+15551234567",
               outbound_min_criticality="High")


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


def _afib_event():
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0
    return process_window(w, _Afib(), MewsVitalsAnalysis())


# --- fakes ---------------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self):
        self.kv = {}

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def get(self, k):
        return self.kv.get(k)

    def delete(self, k):
        self.kv.pop(k, None)


class _State:
    def __init__(self):
        self.authenticated = False
        self.patient_ref = None


class _FakeWorking:
    def __init__(self):
        self.states = {}
        self.turns = {}

    def get_or_create(self, sid):
        return self.states.setdefault(sid, _State())

    def append_turn(self, sid, turn):
        self.turns.setdefault(sid, []).append(turn)

    def set_authenticated(self, sid, *, patient_ref=None, event_ref=None):
        s = self.get_or_create(sid)
        s.authenticated = True
        if patient_ref:
            s.patient_ref = patient_ref
        if event_ref:
            s.event_ref = event_ref


class _Result:
    def __init__(self, answer, declined=False):
        self.answer = answer
        self.declined = declined


class _FakeOrch:
    def __init__(self):
        self.calls = []

    def handle_turn(self, sid, text):
        self.calls.append((sid, text))
        return _Result(f"answer:{text}")


class _NoConverseOrch:
    def handle_turn(self, *a, **k):
        raise AssertionError("live audio must not script a conversation in the relay")


# --- alert store ---------------------------------------------------------------------------


def test_alert_store_roundtrip():
    store = OutboundAlertStore(_FakeRedis())
    alert = OutboundAlert("room1", "PT1", "E1", "Alert for PT1...", "ICU/1")
    store.put(alert)
    assert store.get("room1") == alert
    store.delete("room1")
    assert store.get("room1") is None


# --- outbound handler ----------------------------------------------------------------------


def _handler(tmp_path, working, orch, *, driver=None):
    from voice.handlers import OutboundHandler

    alert = OutboundAlert("room1", "PT1155", "E1", "Alert: AFib, High criticality.", "ICU/1")
    h = OutboundHandler(orch, working, alert, driver=driver, audit=AuditLog(tmp_path / "a.jsonl"))
    return h, alert


def test_outbound_handler_pin_then_alert_then_followup(tmp_path):
    working, orch = _FakeWorking(), _FakeOrch()
    h, alert = _handler(tmp_path, working, orch)

    assert "PIN" in h.greeting()
    # PHI refused before auth
    assert h.respond("tell me about the patient", session_id="room1").startswith("Please say or enter")
    # correct PIN -> speak THIS event's alert (not the generic greeting) + bind the patient
    assert h.respond("one two three four", session_id="room1") == alert.spoken_alert
    assert working.get_or_create("room1").patient_ref == "PT1155"
    # ...and the called event: "this event" is the alerted one, not the patient's latest (a VF call
    # answered vitals from a newer VT for the same patient, 2026-10-13)
    assert working.get_or_create("room1").event_ref == "E1"
    # the alert is a turn in the conversation, so "can you repeat?" re-voices it (2026-10-28 call)
    assert [t.text for t in working.turns["room1"]] == [alert.spoken_alert]
    # a follow-up routes to the orchestrator, scoped to the session
    assert h.respond("what were the vitals", session_id="room1") == "answer:what were the vitals"
    assert orch.calls == [("room1", "what were the vitals")]


def test_outbound_handler_ack_sets_event_status(tmp_path, monkeypatch):
    import kb.graph.events as events

    recorded = {}
    monkeypatch.setattr(events, "set_event_status",
                        lambda d, uuid, status: recorded.update(uuid=uuid, status=status))

    working, orch = _FakeWorking(), _FakeOrch()
    h, alert = _handler(tmp_path, working, orch, driver=object())
    h.respond("one two three four", session_id="room1")  # authenticate
    reply = h.respond("yes I acknowledge", session_id="room1")

    assert "acknowledg" in reply.lower()
    assert recorded == {"uuid": "E1", "status": "acknowledged"}
    # the call ends after "Goodbye" (2026-10-17: the line stayed open for more queries)
    assert h.is_ended("room1") and not h.is_locked("room1")


def test_outbound_handler_no_or_question_does_not_ack(tmp_path, monkeypatch):
    import kb.graph.events as events

    recorded = []
    monkeypatch.setattr(events, "set_event_status", lambda d, uuid, status: recorded.append(status))

    working, orch = _FakeWorking(), _FakeOrch()
    h, _ = _handler(tmp_path, working, orch, driver=object())
    h.respond("one two three four", session_id="room1")
    h.respond("No, that is not correct", session_id="room1")
    h.respond("Can you confirm the potassium?", session_id="room1")
    assert h.respond("Yeah", session_id="room1", spoken=True) == ""  # backchannel over the alert
    assert recorded == []


def test_pin_lockout_refuses_the_right_pin_afterwards(tmp_path):
    working, orch = _FakeWorking(), _FakeOrch()
    h, _ = _handler(tmp_path, working, orch)
    for _ in range(3):
        reply = h.respond("nine nine nine nine", session_id="room1")
    assert "Ending the call" in reply and h.is_locked("room1")
    assert "Ending the call" in h.respond("one two three four", session_id="room1")
    assert not working.get_or_create("room1").authenticated


# --- worker handler selection --------------------------------------------------------------


def test_resolve_handler_picks_outbound_when_alert_present(monkeypatch):
    import voice.livekit_agent as la

    built = {}
    monkeypatch.setattr(la, "build_outbound_handler",
                        lambda alert, **k: (built.update(k), (f"OUT:{alert.event_id}", "g", None))[1])
    monkeypatch.setattr(la, "build_handler", lambda mode, **k: ("IN", "g2", None))

    class _Store:
        def __init__(self, alert):
            self._a = alert
            self.deleted = []
            self.redis = object()

        def get(self, room):
            return self._a

        def delete(self, room):
            self.deleted.append(room)

    alert = OutboundAlert("room1", "PT1", "E1", "x")
    store = _Store(alert)
    h, _, _ = la.resolve_handler("room1", store)
    assert h == "OUT:E1" and store.deleted == ["room1"]  # alert consumed
    assert built["on_answered"].__name__ == "mark_answered"  # the PIN marks the queued call answered

    assert la.resolve_handler("room2", _Store(None))[0] == "IN"  # no alert -> inbound
    assert la.resolve_handler("room3", None)[0] == "IN"  # no store -> inbound


# --- run_outbound live mode ----------------------------------------------------------------


def test_run_outbound_live_places_call_without_scripting(tmp_path):
    from orchestrator.outbound_flow import run_outbound

    res = run_outbound(
        _afib_event(), driver=object(), orchestrator=_NoConverseOrch(),
        caller=SimulatedCaller([CallOutcome.ANSWERED]), utterances=["yes"], config=_CFG,
        bed="ICU/1", live_audio=True, sleep_fn=lambda _s: None, audit=AuditLog(tmp_path / "a.jsonl"),
    )
    assert res.called and res.outcome == "answered" and res.status == "reported"
    assert res.transcript and "Alert for patient" in res.transcript[0]


# --- consumer live path --------------------------------------------------------------------


def test_process_bus_event_live_stages_alert_and_places_call(monkeypatch):
    monkeypatch.setattr(bus_consumer, "ensure_patient", lambda d, b, pid: ("ICU", "3"))
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: None)  # new event
    monkeypatch.setattr(bus_consumer, "process_device_event",
                        lambda ev, d, v, bed=None, config=None: None)

    ev = _afib_event()
    store = OutboundAlertStore(_FakeRedis())
    res = bus_consumer.process_bus_event(
        event_to_dict(ev), driver=object(), vector=object(), orchestrator=_NoConverseOrch(),
        beds=object(), utterances=["yes"], channel="voice", config=_CFG,
        caller_factory=lambda room: SimulatedCaller([CallOutcome.ANSWERED]), alert_store=store,
    )
    assert res.called and res.outbound.status == "reported"
    room = f"rmsai-outbound-{ev.window.event_id}"
    staged = store.get(room)
    assert staged is not None
    assert staged.patient_ref == ev.window.patient_ref and staged.event_id == ev.window.event_id
    assert "Alert for patient" in staged.spoken_alert


# --- which event an acknowledgment closes (live call 2026-10-14) ----------------------------------

def _vf_call(tmp_path, monkeypatch):
    import kb.graph.events as events
    from voice.handlers import OutboundHandler

    recorded = []
    monkeypatch.setattr(events, "set_event_status", lambda d, uuid, status: recorded.append(uuid))
    alert = OutboundAlert("room1", "PT998224", "E-VF", "Alert…", "Unit1/Unit1-Bed02",
                          event_type="VENTRICULAR_FIBRILLATION")
    h = OutboundHandler(_FakeOrch(), _FakeWorking(), alert, driver=object(),
                        audit=AuditLog(tmp_path / "a.jsonl"))
    h.respond("one two three four", session_id="room1")
    return h, recorded


def test_ack_confirmation_names_the_called_event(tmp_path, monkeypatch):
    h, recorded = _vf_call(tmp_path, monkeypatch)
    out = h.respond("Acknowledge this event", session_id="room1")
    assert "ventricular fibrillation alert for patient PT998224 on bed 2" in out
    assert recorded == ["E-VF"]


def test_ack_naming_another_bed_is_refused_not_applied_to_this_call(tmp_path, monkeypatch):
    # The call also discussed bed 7; "acknowledge bed seven" must not close bed 2's VF.
    h, recorded = _vf_call(tmp_path, monkeypatch)
    out = h.respond("acknowledge the event on bed seven", session_id="room1")
    assert recorded == [] and "only acknowledge" in out and "in the app" in out
    out = h.respond("acknowledge PT986685", session_id="room1")
    assert recorded == []


def test_ack_naming_this_bed_or_patient_is_applied(tmp_path, monkeypatch):
    h, recorded = _vf_call(tmp_path, monkeypatch)
    h.respond("acknowledge bed two", session_id="room1")
    assert recorded == ["E-VF"]


def test_scripted_confirm_back_names_the_event():
    from types import SimpleNamespace

    from orchestrator.outbound_flow import _confirm_back

    ev = SimpleNamespace(event_type="VENTRICULAR_FIBRILLATION",
                         window=SimpleNamespace(patient_ref="PT998224"))
    assert _confirm_back(ev, "Unit1/Unit1-Bed02") == (
        "I heard you acknowledge the ventricular fibrillation alert for patient PT998224 on bed 2 "
        "— is that correct?")


def test_a_plain_no_keeps_the_alert_open_and_says_so(tmp_path, monkeypatch):
    h, recorded = _vf_call(tmp_path, monkeypatch)
    out = h.respond("No, that is not correct", session_id="room1")
    assert recorded == [] and "not acknowledged" in out and "stays open" in out
    # A question containing a negation is still a question, not a "no".
    assert h.respond("why don't we check the vitals", session_id="room1").startswith("answer:")
