"""P4: the phone worker registers on the telephony LiveKit as its own agent; the app worker doesn't."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from common.config import Config
from voice.livekit_agent import worker_config

_LOCAL = Config(livekit_url="ws://localhost:7880", livekit_api_key="devkey",
                livekit_api_secret="local-secret")
_SPLIT = replace(_LOCAL, livekit_sip_url="wss://voiceapptest.livekit.cloud",
                 livekit_sip_api_key="APIcloud", livekit_sip_api_secret="cloud-secret")
_COMPOSE = Path(__file__).resolve().parents[1] / "infra" / "docker-compose.yml"


def test_app_role_is_the_config_itself():
    assert worker_config("app", _SPLIT) is _SPLIT


def test_phone_role_registers_on_telephony_server_as_phone_agent():
    cfg = worker_config("phone", _SPLIT)
    assert cfg.livekit_url == "wss://voiceapptest.livekit.cloud"
    assert (cfg.livekit_api_key, cfg.livekit_api_secret) == ("APIcloud", "cloud-secret")
    assert cfg.livekit_agent_name == "rmsai-agent-phone"
    assert cfg.livekit_worker_http_port == 8082
    assert cfg.livekit_redispatch_on_start is False  # inbox rooms don't exist on Cloud


def test_phone_role_refused_in_single_server_mode():
    # would register a second `rmsai-agent` on the same server and split dispatches
    with pytest.raises(ValueError, match="LIVEKIT_SIP_URL"):
        worker_config("phone", _LOCAL)


def test_unknown_role_refused():
    with pytest.raises(ValueError, match="VOICE_WORKER_ROLE"):
        worker_config("both", _SPLIT)


def test_worker_options_use_the_phone_identity():
    pytest.importorskip("livekit.agents")
    from voice.livekit_agent import build_worker_options

    opts = build_worker_options(worker_config("phone", _SPLIT))
    assert opts.agent_name == "rmsai-agent-phone"
    assert opts.ws_url == "wss://voiceapptest.livekit.cloud"
    assert opts.port == 8082


def test_compose_phone_worker_is_profiled_and_role_pinned():
    svc = yaml.safe_load(_COMPOSE.read_text())["services"]
    phone, app = svc["voice-worker-phone"], svc["voice-worker"]
    assert phone["profiles"] == ["telephony"]  # a plain `up` never starts it
    assert phone["environment"]["VOICE_WORKER_ROLE"] == "phone"
    assert "livekit" not in phone["depends_on"]  # it talks to Cloud, not the local server
    assert phone["command"] == app["command"]  # same worker, different role
    # the two health ports never collide under host networking
    assert "8082" in phone["environment"]["LIVEKIT_SIP_WORKER_HTTP_PORT"]
    assert "8091" in app["environment"]["LIVEKIT_WORKER_HTTP_PORT"]


def test_cli_connection_env_is_pinned_to_the_worker_server():
    # livekit-agents' CLI reads LIVEKIT_URL/API_KEY/API_SECRET from the environment and overrides
    # WorkerOptions; .env holds the app server there, so the phone worker must re-pin them.
    from voice.livekit_agent import pin_cli_connection_env

    env = {"LIVEKIT_URL": "ws://localhost:7880", "LIVEKIT_API_KEY": "devkey",
           "LIVEKIT_API_SECRET": "local-secret"}
    pin_cli_connection_env(worker_config("phone", _SPLIT), env)
    assert env == {"LIVEKIT_URL": "wss://voiceapptest.livekit.cloud", "LIVEKIT_API_KEY": "APIcloud",
                   "LIVEKIT_API_SECRET": "cloud-secret"}
