"""Phase 9 Step 2: the gateway `POST /session` PIN gate + static worklist app.

Fail-closed: only a correct PIN mints an inbox join token, and the token is scoped to this
facility's inbox room with join/subscribe grants. Every attempt is audited. The static app is
served at `/`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from common.config import DEFAULT  # noqa: E402
from live.gateway import create_app  # noqa: E402

_CFG = replace(
    DEFAULT, hospital_id="h1", inbound_auth_pin="1234",
    livekit_url="ws://lk:7880", livekit_api_key="devkey", livekit_api_secret="devsecret",
)


def _decode_jwt_payload(token: str) -> dict:
    seg = token.split(".")[1]
    seg += "=" * (-len(seg) % 4)  # restore base64 padding
    return json.loads(base64.urlsafe_b64decode(seg))


def _client(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    return TestClient(create_app(_CFG, audit=audit)), audit


def test_correct_pin_mints_scoped_inbox_token(tmp_path):
    client, audit = _client(tmp_path)
    res = client.post("/session", json={"pin": "1234"})
    assert res.status_code == 200
    body = res.json()
    assert body["room"] == "rmsai-inbox-h1"
    assert body["url"] == "ws://lk:7880"
    assert body["identity"].startswith("clinician-")

    grant = _decode_jwt_payload(body["token"])["video"]
    assert grant["room"] == "rmsai-inbox-h1"
    assert grant["roomJoin"] is True and grant["canSubscribe"] is True

    lines = audit.read_all()
    assert lines and lines[-1]["action"] == "inbox_session" and lines[-1]["outcome"] == "authorized"
    # the PIN itself is never recorded
    assert "1234" not in json.dumps(lines[-1])


def test_spoken_word_pin_also_verifies(tmp_path):
    client, _ = _client(tmp_path)
    res = client.post("/session", json={"pin": "one two three four"})
    assert res.status_code == 200 and res.json()["room"] == "rmsai-inbox-h1"


def test_wrong_pin_is_refused_and_audited(tmp_path):
    client, audit = _client(tmp_path)
    res = client.post("/session", json={"pin": "9999"})
    assert res.status_code == 401
    assert "token" not in res.json()
    assert audit.read_all()[-1]["outcome"] == "denied"


def test_unconfigured_livekit_returns_503(tmp_path):
    cfg = replace(_CFG, livekit_api_key="", livekit_api_secret="")
    client = TestClient(create_app(cfg, audit=AuditLog(str(tmp_path / "a.jsonl"))))
    assert client.post("/session", json={"pin": "1234"}).status_code == 503


def test_root_serves_worklist_app(tmp_path):
    client, _ = _client(tmp_path)
    res = client.get("/")
    assert res.status_code == 200 and "rmsai worklist" in res.text


def test_session_url_defaults_to_internal_livekit_url(tmp_path):
    # No LIVEKIT_PUBLIC_URL ⇒ browser gets the internal URL (identical to pre-deploy behavior).
    client, _ = _client(tmp_path)
    assert client.post("/session", json={"pin": "1234"}).json()["url"] == "ws://lk:7880"


def test_session_url_uses_public_livekit_url_when_set(tmp_path):
    # Behind the public edge, the browser gets the public ingress URL; server API still uses internal.
    cfg = replace(_CFG, livekit_public_url="wss://livekit.example.com")
    client = TestClient(create_app(cfg, audit=AuditLog(str(tmp_path / "a.jsonl"))))
    assert client.post("/session", json={"pin": "1234"}).json()["url"] == "wss://livekit.example.com"


def test_session_dispatches_agent_into_inbox_room(tmp_path):
    # A successful PIN must explicitly dispatch the voice agent into the inbox room, so in-app Q&A
    # reaches the worker regardless of start order.
    dispatched: list[str] = []
    client = TestClient(create_app(
        _CFG, audit=AuditLog(str(tmp_path / "a.jsonl")), dispatcher=dispatched.append))
    res = client.post("/session", json={"pin": "1234"})
    assert res.status_code == 200
    assert dispatched == ["rmsai-inbox-h1"]


def test_session_not_dispatched_on_bad_pin(tmp_path):
    # Fail-closed: a wrong PIN mints no token AND dispatches no agent.
    dispatched: list[str] = []
    client = TestClient(create_app(
        _CFG, audit=AuditLog(str(tmp_path / "a.jsonl")), dispatcher=dispatched.append))
    assert client.post("/session", json={"pin": "9999"}).status_code == 401
    assert dispatched == []


def test_session_survives_dispatch_failure(tmp_path):
    # A dispatch hiccup must not fail the session (worklist/artifacts work without the agent).
    def _boom(room):
        raise RuntimeError("livekit down")

    client = TestClient(create_app(
        _CFG, audit=AuditLog(str(tmp_path / "a.jsonl")), dispatcher=_boom))
    res = client.post("/session", json={"pin": "1234"})
    assert res.status_code == 200 and "token" in res.json()
