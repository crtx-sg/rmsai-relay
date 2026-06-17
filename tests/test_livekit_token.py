"""cli.livekit_token: mint a join token for the browser WebRTC test (offline)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import cli.livekit_token as tok
from common.config import DEFAULT

_CFG = replace(DEFAULT, livekit_url="wss://proj.livekit.cloud",
               livekit_api_key="APItest", livekit_api_secret="s3cr3t-s3cr3t-s3cr3t")


def test_token_cli_prints_join_info(capsys, monkeypatch):
    monkeypatch.setattr(tok.Config, "from_env", staticmethod(lambda: _CFG))
    rc = tok.main(["--room", "rmsai-call-demo"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "wss://proj.livekit.cloud" in out
    assert "rmsai-call-demo" in out
    # the printed token is a 3-segment JWT
    token_line = next(l for l in out.splitlines() if l.startswith("Token:"))
    assert token_line.split("Token:")[1].strip().count(".") == 2


def test_token_cli_requires_config(monkeypatch):
    monkeypatch.setattr(tok.Config, "from_env", staticmethod(lambda: DEFAULT))  # no key/secret
    with pytest.raises(SystemExit):
        tok.main(["--room", "r"])
