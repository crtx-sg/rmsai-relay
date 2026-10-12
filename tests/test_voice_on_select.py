"""Companion-app voice on select: muted by default when calls go out alongside the app.

In app+call mode the clinician may be on the phone alert for the very event they select in the app;
the app speaking the report too is double speech. The app sends "/select-silent" when muted, so the
worker scopes the chat without running TTS.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from common.config import DEFAULT
from live.gateway import speak_on_select_default
from voice.handlers import InboxHandler


@pytest.mark.parametrize("mode, enabled, expected", [
    ("app+call", True, False),   # phone alerts alongside the app: muted by default
    ("call+app", True, False),
    ("app", True, True),
    ("call", True, True),        # no app worklist; the default is moot but stays on
    ("app", False, False),       # INBOX_SPEAK_ON_SELECT=false turns it off everywhere
])
def test_default_follows_dispatch_mode(mode, enabled, expected):
    cfg = replace(DEFAULT, dispatch_mode=mode, inbox_speak_on_select=enabled)
    assert speak_on_select_default(cfg) is expected


def test_session_reports_the_default(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from common.audit import AuditLog
    from live.gateway import create_app

    cfg = replace(DEFAULT, hospital_id="h1", inbound_auth_pin="1234", livekit_url="ws://lk:7880",
                  livekit_api_key="devkey", livekit_api_secret="devsecret",
                  dispatch_mode="app+call", inbox_speak_on_select=True)
    client = TestClient(create_app(cfg, audit=AuditLog(str(tmp_path / "a.jsonl"))))
    body = client.post("/session", json={"pin": "1234"}).json()
    assert body["speak_on_select"] is False and body["dispatch_mode"] == "app+call"


class _Mem:
    def __init__(self):
        self.bound = []

    def set_authenticated(self, session_id, **kw):
        self.bound.append((session_id, kw))


@pytest.mark.parametrize("message", ["/select evt-1", "/select-silent evt-1"])
def test_both_select_forms_scope_the_chat_and_reply_nothing(message):
    mem = _Mem()
    h = InboxHandler(SimpleNamespace(), mem)
    assert h.respond(message, session_id="s1") == ""
    assert mem.bound == [("s1", {"patient_ref": "evt-1", "event_ref": "evt-1"})]
    assert h._selected["s1"] == ("evt-1", "evt-1")
