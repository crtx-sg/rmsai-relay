"""E5 server side: `POST /metrics`, `POST /event-info`, and the worklist-row extras."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from common.config import DEFAULT  # noqa: E402
from live import gateway  # noqa: E402
from live.gateway import create_app  # noqa: E402
from live.inbox import build_event_message  # noqa: E402
from orchestrator.bus_consumer import _row_extras  # noqa: E402
from voice.livekit_cloud import access_token  # noqa: E402

_CFG = replace(DEFAULT, hospital_id="h1", inbound_auth_pin="1234", livekit_url="ws://lk:7880",
               livekit_api_key="devkey", livekit_api_secret="devsecret")
VT, NSR = "VENTRICULAR_TACHYCARDIA", "NORMAL_SINUS"


def _row(eid, pred, truth, model="ecg_transconv:v2", gate=True, **kw):
    r = {"patient": "PT992591", "id": eid, "timestamp": 1.0, "processed_at": 2.0,
         "event_type": pred, "confidence": 0.5, "ground_truth_condition": truth,
         "criticality": "Critical", "model_id": model, "model_classes": None,
         "eval_outcome": "TP", "alert_gate": gate, "alert_reason_code": "vitals_alert",
         "why": "Shown because…", "delivered_app": False, "source_kind": "ecg_sigma",
         "source_dataset": "incart", "source_record": "incart:I05", "source_sample": 263874,
         "source_split": "test"}
    r.update(kw)
    return r


@pytest.fixture()
def graph(monkeypatch):
    seen = {}

    def fake_eval(driver, **kw):
        seen["filters"] = kw
        return [_row("e1", VT, VT), _row("e2", NSR, NSR, gate=False, eval_outcome="TN")]

    def fake_info(driver, uuid):
        if uuid == "unknown":
            return None
        return {**_row(uuid, VT, VT), "why_json": json.dumps({"headline": "Shown because…",
                                                              "basis": "vitals"}),
                "eval_unscorable_reason": None, "source_label_method": "rhythm_annotation",
                "source_device": "RMSAI-SimDevice-v2.0", "delivered_call": "answered"}

    monkeypatch.setattr(gateway, "eval_events", fake_eval)
    monkeypatch.setattr(gateway, "get_event_info", fake_info)
    return seen


def _client(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    return TestClient(create_app(_CFG, audit=audit, driver=object())), audit


def _token(room="rmsai-inbox-h1"):
    return access_token(identity="clinician-x", room=room, config=_CFG)


def test_metrics_returns_per_model_summary_and_every_labelled_event(tmp_path, graph):
    client, audit = _client(tmp_path)
    res = client.post("/metrics", json={"session": _token(), "since": "24h", "dataset": "incart"})
    assert res.status_code == 200
    body = res.json()
    assert [m["model_id"] for m in body["models"]] == ["ecg_transconv:v2"]
    s = body["models"][0]["summary"]
    assert s["counts"]["TP"] == 1 and s["counts"]["TN"] == 1
    assert body["models"][0]["sources"] == ["incart"]
    # not-alerted events are listed too (the TN never reached the worklist)
    assert [(e["event_id"], e["alert"]) for e in body["events"]] == [("e1", True), ("e2", False)]
    assert body["events"][0]["source"] == "INCART I05 @263874 (test split)"
    assert graph["filters"]["dataset"] == "incart" and graph["filters"]["since"] is not None
    assert audit.read_all()[-1]["action"] == "view_metrics"


def test_metrics_requires_a_valid_session(tmp_path, graph):
    client, audit = _client(tmp_path)
    assert client.post("/metrics", json={"session": "bad"}).status_code == 401
    assert client.post("/metrics", json={"session": _token("rmsai-inbox-other")}).status_code == 401
    assert "filters" not in graph and audit.read_all()[-1]["outcome"] == "unauthorized"


def test_metrics_rejects_a_bad_since(tmp_path, graph):
    client, _ = _client(tmp_path)
    assert client.post("/metrics", json={"session": _token(), "since": "soon"}).status_code == 400


def test_event_info_explains_one_event(tmp_path, graph):
    client, audit = _client(tmp_path)
    res = client.post("/event-info", json={"session": _token(), "event_id": "e1"})
    assert res.status_code == 200
    b = res.json()
    assert b["why"] == "Shown because…" and b["explanation"]["basis"] == "vitals"
    assert b["source"] == "INCART I05 @263874 (test split)"
    assert b["provenance"]["record"] == "incart:I05" and b["provenance"]["label_method"] == \
        "rhythm_annotation"
    assert b["delivery"] == {"app": False, "call": "answered"}
    assert (b["truth"], b["outcome"], b["model_id"]) == (VT, "TP", "ecg_transconv:v2")
    assert audit.read_all()[-1]["subject"] == "PT992591"


def test_event_info_unknown_and_unauthorized(tmp_path, graph):
    client, _ = _client(tmp_path)
    assert client.post("/event-info", json={"session": _token(), "event_id": "unknown"}
                       ).status_code == 404
    assert client.post("/event-info", json={"session": "bad", "event_id": "e1"}).status_code == 401


def test_row_extras_source_badge_only_for_demo_data():
    base = {"why": "Shown because…", "truth": VT, "outcome": "TP", "source": "MIT-BIH 207"}
    assert _row_extras({**base, "source_kind": "ecg_sigma"}) == {
        "why": "Shown because…", "source": "MIT-BIH 207", "truth": VT, "outcome": "TP"}
    dev = _row_extras({**base, "source_kind": "device", "truth": None})
    assert dev == {"why": "Shown because…"}  # no device badge, no outcome without ground truth
    assert _row_extras(None) == {}


def test_inbox_message_carries_the_extras():
    msg = build_event_message(event_id="e1", patient="PT992591", unit="U1", bed="B1",
                              event_type=VT, ts=1.0, criticality="Critical", status="reported",
                              links={}, why="Shown because…", source="INCART I05", truth=VT,
                              outcome="TP")
    assert (msg["why"], msg["source"], msg["truth"], msg["outcome"]) == (
        "Shown because…", "INCART I05", VT, "TP")
