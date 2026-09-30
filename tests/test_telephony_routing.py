"""P3: phone-call paths use the telephony LiveKit; WebRTC and the app stay on the app's LiveKit."""

from __future__ import annotations

from dataclasses import replace

import pytest

import voice.livekit_cloud as lk_cloud
from cli.call import main as call_main
from cli.consume import livekit_voice_wiring
from common.config import Config
from voice.outbound import LiveKitCaller, SimulatedCaller

_LOCAL = Config(livekit_url="ws://localhost:7880", livekit_api_key="devkey",
                livekit_api_secret="local-secret", outbound_call_number="+15550002000",
                livekit_sip_trunk_id="ST_cloud")
_SPLIT = replace(_LOCAL, livekit_sip_url="wss://voiceapptest.livekit.cloud",
                 livekit_sip_api_key="APIcloud", livekit_sip_api_secret="cloud-secret")


@pytest.fixture
def dispatches(monkeypatch):
    """Capture every create_agent_dispatch as (room, server url, agent name). Never hits LiveKit."""
    seen: list[tuple[str, str, str]] = []

    def fake(room, *, config, agent_name=None, metadata=""):
        seen.append((room, config.livekit_url, agent_name or config.livekit_agent_name))
        return True

    monkeypatch.setattr(lk_cloud, "create_agent_dispatch", fake)
    return seen


def test_sip_dial_and_dispatch_go_to_the_telephony_server(dispatches):
    caller_factory, dispatch_fn = livekit_voice_wiring(_SPLIT, "sip")
    caller = caller_factory("rmsai-outbound-e1")
    assert isinstance(caller, LiveKitCaller) and caller.room == "rmsai-outbound-e1"
    assert caller.config.livekit_url == "wss://voiceapptest.livekit.cloud"
    assert caller.config.livekit_api_key == "APIcloud"
    dispatch_fn("rmsai-outbound-e1")
    assert dispatches == [("rmsai-outbound-e1", "wss://voiceapptest.livekit.cloud",
                           "rmsai-agent-phone")]


def test_webrtc_stays_on_the_app_server(dispatches):
    caller_factory, dispatch_fn = livekit_voice_wiring(_SPLIT, "webrtc")
    assert isinstance(caller_factory("rmsai-outbound-e1"), SimulatedCaller)  # no phone leg
    dispatch_fn("rmsai-outbound-e1")
    assert dispatches == [("rmsai-outbound-e1", "ws://localhost:7880", "rmsai-agent")]


def test_single_server_sip_is_unchanged(dispatches):
    caller_factory, dispatch_fn = livekit_voice_wiring(_LOCAL, "sip")
    assert caller_factory("r").config is _LOCAL
    dispatch_fn("r")
    assert dispatches == [("r", "ws://localhost:7880", "rmsai-agent")]


def test_on_demand_call_dispatches_phone_agent_on_cloud(dispatches, tmp_path, capsys):
    cfg = replace(_SPLIT, audit_log_path=str(tmp_path / "audit.jsonl"))
    rc = call_main(["--caller", "simulated", "--call-id", "t1"], config=cfg)
    assert rc == 0
    assert dispatches == [("rmsai-call-t1", "wss://voiceapptest.livekit.cloud", "rmsai-agent-phone")]


def test_on_demand_call_refuses_partial_telephony_config(tmp_path):
    cfg = replace(_SPLIT, livekit_sip_api_secret="", audit_log_path=str(tmp_path / "a.jsonl"))
    with pytest.raises(SystemExit) as exc:
        call_main(["--caller", "simulated"], config=cfg)
    assert exc.value.code == 2
