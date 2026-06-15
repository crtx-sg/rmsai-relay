"""Inbound text interaction (POC option): PIN-gated, grounded — the text twin of inbound voice."""

from __future__ import annotations

import uuid

import pytest

from common.config import DEFAULT

pytestmark = pytest.mark.infra


def _backends_or_skip():
    from kb.graph.driver import GraphDriver

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
        d.close()
        import redis
        redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2).ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend unreachable: {exc}")


def test_text_chat_pin_gate_then_grounded(monkeypatch, capsys):
    _backends_or_skip()
    from cli.text_chat import main

    sid = f"text-{uuid.uuid4().hex[:8]}"
    lines = iter([
        "what is the rate control for atrial fibrillation",  # pre-auth -> refused
        "1234",                                              # authenticate (default PIN)
        "what is the first-line management for atrial fibrillation",  # grounded
        "quit",
    ])

    def fake_input(prompt: str = "") -> str:
        return next(lines)

    monkeypatch.setattr("builtins.input", fake_input)
    rc = main(["--session", sid, "--embedder", "hashing"])
    out = capsys.readouterr().out

    assert rc == 0
    # the pre-auth question is refused (PHI gated), the post-auth one is answered (grounded)
    assert "PIN" in out
    assert "authenticated" in out.lower()
    assert "grounded answer" in out
