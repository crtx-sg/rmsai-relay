"""Phase 9 Step 3: acknowledge round-trip via `POST /ack`.

An app ack (with a valid inbox session token) flips `MonitoredEvent.status -> acknowledged`, audits
it (against the patient pseudonym, `surface="app"`), and pushes a `status` message back to the inbox
so every surface reflects it. Fail-closed: a bad/expired/wrong-room token is refused (401); an
unknown event is refused (404). The graph writes are monkeypatched; the inbox publisher's send is a
capturing fake.
"""

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
from live.inbox import InboxPublisher  # noqa: E402
from voice.livekit_cloud import access_token, verify_access_token  # noqa: E402

_CFG = replace(
    DEFAULT, hospital_id="h1", inbound_auth_pin="1234",
    livekit_url="ws://lk:7880", livekit_api_key="devkey", livekit_api_secret="devsecret",
)


@pytest.fixture()
def graph(monkeypatch):
    """Fake the two graph functions the gateway calls; record flips + which events 'exist'."""
    calls = {"flips": []}
    monkeypatch.setattr(gateway, "get_event_patient",
                        lambda drv, uuid: None if uuid == "unknown" else "PT1155")
    monkeypatch.setattr(gateway, "set_event_status",
                        lambda drv, uuid, status: calls["flips"].append((uuid, status)))
    return calls


def _client(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    captured: list[tuple[str, dict]] = []
    pub = InboxPublisher("rmsai-inbox-h1",
                         send_fn=lambda room, data: captured.append((room, json.loads(data))))
    client = TestClient(create_app(_CFG, audit=audit, driver=object(), publisher=pub))
    return client, audit, captured


def _token(room="rmsai-inbox-h1"):
    return access_token(identity="clinician-x", room=room, config=_CFG)


def test_ack_flips_status_audits_and_pushes_back(tmp_path, graph):
    client, audit, captured = _client(tmp_path)
    res = client.post("/ack", json={"event_id": "evt-1", "session": _token()})

    assert res.status_code == 200 and res.json() == {"event_id": "evt-1", "status": "acknowledged"}
    assert graph["flips"] == [("evt-1", "acknowledged")]

    line = audit.read_all()[-1]
    assert line["action"] == "acknowledgment" and line["outcome"] == "acknowledged"
    assert line["subject"] == "PT1155" and line["extra"]["surface"] == "app"

    assert captured == [("rmsai-inbox-h1",
                         {"type": "status", "event_id": "evt-1", "status": "acknowledged"})]


def test_ack_with_bad_token_is_refused(tmp_path, graph):
    client, audit, captured = _client(tmp_path)
    res = client.post("/ack", json={"event_id": "evt-1", "session": "not.a.token"})
    assert res.status_code == 401
    assert graph["flips"] == [] and captured == []
    assert audit.read_all()[-1]["outcome"] == "unauthorized"


def test_ack_with_wrong_room_token_is_refused(tmp_path, graph):
    client, _, captured = _client(tmp_path)
    res = client.post("/ack", json={"event_id": "evt-1", "session": _token("rmsai-inbox-other")})
    assert res.status_code == 401
    assert graph["flips"] == [] and captured == []


def test_ack_unknown_event_is_404(tmp_path, graph):
    client, audit, captured = _client(tmp_path)
    res = client.post("/ack", json={"event_id": "unknown", "session": _token()})
    assert res.status_code == 404
    assert graph["flips"] == [] and captured == []
    assert audit.read_all()[-1]["outcome"] == "unknown_event"


def test_verify_access_token_rejects_tampered_and_expired():
    token = access_token(identity="c", room="rmsai-inbox-h1", config=_CFG, ttl_seconds=100, now=1000)
    assert verify_access_token(token, _CFG, now=1050) is not None
    assert verify_access_token(token, _CFG, now=2000) is None  # expired
    assert verify_access_token(token + "x", _CFG, now=1050) is None  # tampered signature
    assert verify_access_token("garbage", _CFG, now=1050) is None
