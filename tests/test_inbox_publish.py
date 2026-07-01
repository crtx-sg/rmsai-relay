"""Phase 9 Step 1: DISPATCH_MODE branch + inbox publisher + scoped artifact tokens.

The criticality gate is real (`should_call` runs); the graph/vector/outbound collaborators are
monkeypatched, and the inbox publisher's LiveKit send is a capturing fake — so we assert the
routing (which surface fires) and the pseudonym-only message schema without any live backend.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from common.config import DEFAULT
from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.serialize import dict_to_event, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from live.artifact_tokens import ArtifactTokenStore
from live.inbox import (
    InboxPublisher,
    _server_api_url,
    artifact_kinds_for,
    build_event_message,
)
from orchestrator import bus_consumer
from orchestrator.outbound_flow import OutboundResult

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_CFG = replace(DEFAULT, outbound_enabled=True, outbound_min_criticality="High", hospital_id="h1")


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


def _event():
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0  # high MEWS -> critical
    return process_window(w, _Afib(), MewsVitalsAnalysis())


class _FakeRedis:
    """In-memory Redis with a virtual clock, so TTL expiry is deterministic."""

    def __init__(self, clock: float = 1000.0) -> None:
        self._d: dict[str, tuple[str, float | None]] = {}
        self.clock = clock

    def set(self, key, val, ex=None):
        self._d[key] = (val, None if ex is None else self.clock + ex)

    def get(self, key):
        item = self._d.get(key)
        if item is None:
            return None
        val, exp = item
        if exp is not None and self.clock >= exp:
            self._d.pop(key, None)
            return None
        return val.encode("utf-8")

    def delete(self, key):
        self._d.pop(key, None)


@pytest.fixture()
def patched(monkeypatch):
    calls = {"voice": [], "text": []}
    monkeypatch.setattr(bus_consumer, "ensure_patient", lambda d, b, pid: ("ICU", "3"))
    monkeypatch.setattr(bus_consumer, "process_device_event",
                        lambda ev, d, v, bed=None, config=None: None)
    monkeypatch.setattr(bus_consumer, "run_outbound",
                        lambda ev, **kw: (calls["voice"].append(kw["bed"])
                                          or OutboundResult(called=True, decision_reason="ok",
                                                            outcome="answered", attempts=1,
                                                            status="acknowledged")))
    monkeypatch.setattr(bus_consumer, "run_text_notify",
                        lambda ev, **kw: (calls["text"].append(kw["to"])
                                          or OutboundResult(called=True, decision_reason="ok",
                                                            channel="text", outcome="delivered",
                                                            status="acknowledged")))
    return calls


def _publisher():
    captured: list[tuple[str, dict]] = []
    pub = InboxPublisher(
        "rmsai-inbox-h1",
        send_fn=lambda room, data: captured.append((room, json.loads(data))),
    )
    return pub, captured


def _run(mode, patched, *, pub=None, store=None):
    return bus_consumer.process_bus_event(
        event_to_dict(_event()), driver=object(), vector=object(), orchestrator=object(),
        beds=object(), utterances=["yes", "yes"], channel="voice",
        config=replace(_CFG, dispatch_mode=mode), inbox_publisher=pub, token_store=store,
    )


# --- dispatch routing --------------------------------------------------------------------------

def test_app_mode_pushes_event_with_scoped_links_and_no_call(patched):
    pub, captured = _publisher()
    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    res = _run("app", patched, pub=pub, store=store)

    assert res.app_dispatched and not res.called
    assert not patched["voice"] and not patched["text"]  # app-only: no phone surface

    assert len(captured) == 1
    room, msg = captured[0]
    assert room == "rmsai-inbox-h1"
    assert msg["type"] == "event" and msg["status"] == "reported"
    assert msg["patient"].startswith("PT") and msg["bed"] == "3" and msg["unit"] == "ICU"
    assert msg["event_type"] == "ATRIAL_FIBRILLATION"
    # every present artifact kind (on the bus-reconstructed event) gets a scoped, verifiable link
    expected_kinds = set(artifact_kinds_for(dict_to_event(event_to_dict(_event())), config=_CFG))
    assert set(msg["links"]) == expected_kinds
    assert msg["artifact_kinds"] == list(msg["links"])
    assert "report" in msg["links"]
    for kind, link in msg["links"].items():
        assert link["url"] == f"/artifact/{link['token']}"
        grant = store.verify(link["token"], kind)
        assert grant is not None and grant.kind == kind and grant.event_id == msg["event_id"]


def test_app_call_mode_pushes_and_calls(patched):
    pub, captured = _publisher()
    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    res = _run("app+call", patched, pub=pub, store=store)

    assert res.app_dispatched and res.called
    assert patched["voice"] == ["ICU/3"]
    assert len(captured) == 1 and captured[0][1]["type"] == "event"


def test_call_mode_never_pushes_to_inbox(patched):
    pub, captured = _publisher()
    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    res = _run("call", patched, pub=pub, store=store)

    assert res.called and not res.app_dispatched
    assert captured == []  # call surface only
    assert patched["voice"] == ["ICU/3"]


# --- publisher + token store units -------------------------------------------------------------

def test_inbox_push_failure_does_not_poison_event(patched):
    # A LiveKit hiccup while pushing must not fail a persisted, critical event.
    class _BoomPublisher:
        room = "rmsai-inbox-h1"

        def publish_event(self, message):
            raise RuntimeError("connection reset by peer")

    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    res = _run("app+call", patched, pub=_BoomPublisher(), store=store)
    assert not res.app_dispatched          # push failed, flagged as not dispatched
    assert res.called                      # ...but the call surface still fired (app+call)
    assert patched["voice"] == ["ICU/3"]


def test_server_api_url_converts_ws_to_http():
    assert _server_api_url("ws://localhost:7880") == "http://localhost:7880"
    assert _server_api_url("wss://x.livekit.cloud") == "https://x.livekit.cloud"
    assert _server_api_url("http://localhost:7880") == "http://localhost:7880"


def test_publish_status_message():
    pub, captured = _publisher()
    pub.publish_status("evt-123", "acknowledged")
    assert captured == [("rmsai-inbox-h1", {"type": "status", "event_id": "evt-123",
                                            "status": "acknowledged"})]


def test_build_event_message_refuses_non_pseudonym():
    with pytest.raises(ValueError, match="pseudonym"):
        build_event_message(event_id="e1", patient="John Doe", unit="ICU", bed="3",
                            event_type="ATRIAL_FIBRILLATION", ts=0.0, criticality="High",
                            status="reported", links={})


def test_artifact_token_roundtrip_unknown_and_kind_mismatch():
    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    token, expires = store.mint("evt-9", "ecg_strip")
    assert expires > 0
    grant = store.verify(token, "ecg_strip")
    assert grant is not None and grant.event_id == "evt-9" and grant.kind == "ecg_strip"
    # kind mismatch (token scoped to ecg_strip can't fetch the report) and unknown token are refused
    assert store.verify(token, "report") is None
    assert store.verify("bogus", "ecg_strip") is None
    assert store.verify("", "ecg_strip") is None


def test_artifact_token_expires():
    redis = _FakeRedis(clock=1000.0)
    store = ArtifactTokenStore(redis, ttl_seconds=300)
    token, _ = store.mint("evt-9", "report")
    assert store.verify(token, "report") is not None
    redis.clock = 1000.0 + 301  # advance past TTL
    assert store.verify(token, "report") is None
