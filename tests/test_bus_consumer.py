"""Bus consumer orchestration: persist always, route on the (real) criticality gate.

Collaborators that need live backends (graph persist, outbound loop, patient bootstrap) are
monkeypatched; the real `should_call` gate runs so the routing decision is genuinely exercised.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from common.config import DEFAULT
from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.serialize import event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator import bus_consumer
from orchestrator.outbound_flow import OutboundResult

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_CFG = replace(DEFAULT, outbound_enabled=True, outbound_min_criticality="High")


def _event(model):
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0
    return process_window(w, model, MewsVitalsAnalysis())


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


class _Normal(ECGModel):
    def predict(self, window):
        return "NORMAL_SINUS", 0.99


@pytest.fixture()
def patched(monkeypatch):
    calls = {"persist": [], "voice": [], "text": []}
    monkeypatch.setattr(bus_consumer, "ensure_patient", lambda d, b, pid: ("ICU", "3"))
    monkeypatch.setattr(bus_consumer, "process_device_event",
                        lambda ev, d, v, bed=None, config=None: calls["persist"].append(ev.window.event_id))

    def _voice(ev, **kw):
        calls["voice"].append(kw["bed"])
        return OutboundResult(called=True, decision_reason="ok", outcome="answered",
                              attempts=1, status="acknowledged")

    def _text(ev, **kw):
        calls["text"].append(kw["to"])
        return OutboundResult(called=True, decision_reason="ok", channel="text",
                              outcome="delivered", status="acknowledged")

    monkeypatch.setattr(bus_consumer, "run_outbound", _voice)
    monkeypatch.setattr(bus_consumer, "run_text_notify", _text)
    return calls


def _run(payload, patched, *, config=_CFG, **kw):
    return bus_consumer.process_bus_event(
        payload, driver=object(), vector=object(), orchestrator=object(), beds=object(),
        utterances=["yes", "yes"], config=config, **kw,
    )


def test_critical_event_persists_and_calls(patched):
    res = _run(event_to_dict(_event(_Afib())), patched, channel="voice")
    assert res.persisted and res.called
    assert patched["persist"] == [res.event_uuid]
    assert patched["voice"] == ["ICU/3"] and not patched["text"]


def test_text_channel_routes_to_notify(patched):
    res = _run(event_to_dict(_event(_Afib())), patched, channel="text", to="+15550009999")
    assert res.called and res.outbound.channel == "text"
    assert patched["text"] == ["+15550009999"] and not patched["voice"]


def test_false_positive_persists_but_never_calls(patched):
    # D10 guard with the vitals-override disabled: a NORMAL_SINUS persists but never dials out.
    cfg = replace(_CFG, criticality_fp_override_on_vitals=False)
    res = _run(event_to_dict(_event(_Normal())), patched, channel="voice", config=cfg)
    assert res.persisted and not res.called
    assert res.decision_reason == "false_positive"
    assert patched["persist"] and not patched["voice"] and not patched["text"]


class _WeakAfib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.40  # < the 60% default gate


def test_uncertain_rhythm_on_a_deteriorating_patient_alerts_on_the_vitals(patched, capsys):
    # HR 145 -> high MEWS. The rhythm can't carry the alert at 40%, so the vitals do: the clinician
    # is still reached, and the console says the alert was re-based rather than asserted.
    res = _run(event_to_dict(_event(_WeakAfib())), patched, channel="voice")
    assert res.called and res.decision_reason.startswith("vitals_alert")
    out = capsys.readouterr().out
    assert "VITALS-DRIVEN ALERT" in out and "rhythm flagged unconfirmed" in out


def test_false_positive_overridden_by_vitals_calls(patched):
    # Same NORMAL_SINUS event (HR=145 -> high MEWS) DOES call when the vitals-override is on.
    res = _run(event_to_dict(_event(_Normal())), patched, channel="voice")
    assert res.persisted and res.called
    assert res.decision_reason.startswith("fp_override")
    assert patched["voice"] == ["ICU/3"]


class _FakeAlertStore:
    def __init__(self):
        self.staged = []

    def put(self, alert):
        self.staged.append(alert.session_id)


def test_livekit_path_dispatches_agent_into_event_room(patched):
    # The live LiveKit path stages the alert AND explicitly dispatches the (named) agent into the
    # per-event room, so the worker is present when the call connects / clinician joins.
    store = _FakeAlertStore()
    dispatched = []
    res = _run(event_to_dict(_event(_Afib())), patched, channel="voice",
               caller_factory=lambda room: object(), dispatch_fn=dispatched.append,
               alert_store=store)
    assert res.called
    room = f"rmsai-outbound-{res.event_uuid}"
    assert dispatched == [room]        # agent requested for the event room
    assert store.staged == [room]      # alert staged for the same room


def test_dispatch_failure_does_not_abort_the_call(patched):
    # Best-effort: a dispatch hiccup must not stop the (already gated + staged) call from proceeding.
    def _boom(room):
        raise RuntimeError("livekit down")

    res = _run(event_to_dict(_event(_Afib())), patched, channel="voice",
               caller_factory=lambda room: object(), dispatch_fn=_boom,
               alert_store=_FakeAlertStore())
    assert res.persisted and res.called
