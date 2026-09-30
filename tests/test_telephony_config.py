"""Telephony split: phone-call paths on a second LiveKit (e.g. Cloud), everything else unchanged."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from common.config import Config

_LOCAL = Config(livekit_url="ws://localhost:7880", livekit_public_url="wss://edge.example",
                livekit_api_key="devkey", livekit_api_secret="local-secret",
                hospital_id="h1", redis_url="redis://localhost:6379/0",
                livekit_sip_trunk_id="ST_abc", outbound_call_number="+15551230000")
_SPLIT = replace(_LOCAL, livekit_sip_url="wss://voiceapptest.livekit.cloud",
                 livekit_sip_api_key="APIcloud", livekit_sip_api_secret="cloud-secret")

# The only fields `telephony()` may change; everything else is shared by both servers' paths.
_SWAPPED = {"livekit_url", "livekit_public_url", "livekit_api_key", "livekit_api_secret",
            "livekit_agent_name", "livekit_worker_http_port", "livekit_redispatch_on_start"}


def test_single_server_mode_is_unchanged():
    assert not _LOCAL.telephony_split
    assert _LOCAL.telephony() is _LOCAL  # calls use livekit_* exactly as before


def test_split_mode_points_call_paths_at_the_telephony_server():
    tel = _SPLIT.telephony()
    assert _SPLIT.telephony_split
    assert tel.livekit_url == "wss://voiceapptest.livekit.cloud"
    assert (tel.livekit_api_key, tel.livekit_api_secret) == ("APIcloud", "cloud-secret")
    assert tel.livekit_agent_name == "rmsai-agent-phone"
    assert tel.livekit_worker_http_port == 8082  # never the app worker's 8081 (host networking)
    assert tel.livekit_redispatch_on_start is False  # inbox rooms exist only on the app server
    assert tel.livekit_public_url == ""


def test_split_mode_shares_everything_else():
    tel = _SPLIT.telephony()
    changed = {f.name for f in fields(Config) if getattr(tel, f.name) != getattr(_SPLIT, f.name)}
    assert changed <= _SWAPPED
    # staged alerts (Redis), the inbox room, the trunk and the numbers are the same on both paths
    assert (tel.redis_url, tel.hospital_id, tel.livekit_sip_trunk_id, tel.outbound_call_number) == (
        _SPLIT.redis_url, _SPLIT.hospital_id, _SPLIT.livekit_sip_trunk_id, _SPLIT.outbound_call_number)


def test_app_side_config_is_not_mutated():
    _SPLIT.telephony()
    assert _SPLIT.livekit_url == "ws://localhost:7880"
    assert _SPLIT.livekit_agent_name == "rmsai-agent"


@pytest.mark.parametrize("missing", ["livekit_sip_api_key", "livekit_sip_api_secret"])
def test_partial_split_config_fails_loud(missing):
    with pytest.raises(ValueError, match=missing.upper()):
        replace(_SPLIT, **{missing: ""}).telephony()


def test_inbound_allow_list_falls_back_to_on_call_number():
    assert _LOCAL.inbound_allowed_numbers == ("+15551230000",)
    explicit = replace(_LOCAL, sip_inbound_allowed_numbers=("+15550001111", "+15550002222"))
    assert explicit.inbound_allowed_numbers == ("+15550001111", "+15550002222")
    assert replace(_LOCAL, outbound_call_number="").inbound_allowed_numbers == ()  # no one


def test_from_env_parses_telephony_keys(monkeypatch):
    env = {
        "LIVEKIT_SIP_URL": "wss://voiceapptest.livekit.cloud",
        "LIVEKIT_SIP_API_KEY": "APIcloud", "LIVEKIT_SIP_API_SECRET": "cloud-secret",
        "TWILIO_SIP_TERMINATION_URI": "rmsai.pstn.twilio.com",
        "TWILIO_SIP_USERNAME": "relay", "TWILIO_SIP_PASSWORD": "pw",
        "SIP_INBOUND_ALLOWED_NUMBERS": "+15550001111, +15550002222",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = Config.from_env()
    assert cfg.telephony_split and cfg.sip_agent_name == "rmsai-agent-phone"
    assert cfg.twilio_sip_termination_uri == "rmsai.pstn.twilio.com"
    assert cfg.sip_inbound_allowed_numbers == ("+15550001111", "+15550002222")
    assert cfg.telephony().livekit_url == "wss://voiceapptest.livekit.cloud"


def test_new_secrets_stay_out_of_repr():
    text = repr(replace(_SPLIT, twilio_sip_password="twilio-pw"))
    assert "cloud-secret" not in text and "twilio-pw" not in text
