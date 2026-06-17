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

    def get_or_create(self, sid):
        return self.states.setdefault(sid, _State())

    def set_authenticated(self, sid, *, patient_ref=None):
        s = self.get_or_create(sid)
        s.authenticated = True
        if patient_ref:
            s.patient_ref = patient_ref


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
    assert "authenticate" in h.respond("tell me about the patient", session_id="room1").lower()
    # correct PIN -> speak THIS event's alert (not the generic greeting) + bind the patient
    assert h.respond("one two three four", session_id="room1") == alert.spoken_alert
    assert working.get_or_create("room1").patient_ref == "PT1155"
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


# --- worker handler selection --------------------------------------------------------------


def test_resolve_handler_picks_outbound_when_alert_present(monkeypatch):
    import voice.livekit_agent as la

    monkeypatch.setattr(la, "build_outbound_handler", lambda alert, **k: (f"OUT:{alert.event_id}", "g", None))
    monkeypatch.setattr(la, "build_handler", lambda mode, **k: ("IN", "g2", None))

    class _Store:
        def __init__(self, alert):
            self._a = alert
            self.deleted = []

        def get(self, room):
            return self._a

        def delete(self, room):
            self.deleted.append(room)

    alert = OutboundAlert("room1", "PT1", "E1", "x")
    store = _Store(alert)
    h, _, _ = la.resolve_handler("room1", store)
    assert h == "OUT:E1" and store.deleted == ["room1"]  # alert consumed

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
    monkeypatch.setattr(bus_consumer, "process_device_event", lambda ev, d, v, bed=None: None)

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
