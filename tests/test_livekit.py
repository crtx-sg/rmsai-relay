"""LiveKit Cloud: access-token generation (offline, verifiable) + config/factory wiring."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import replace

import pytest

from common.config import DEFAULT
from voice.livekit_cloud import access_token, is_configured
from voice.outbound import CallOutcome, LiveKitCaller, get_caller

_CFG = replace(DEFAULT, livekit_url="wss://demo.livekit.cloud",
               livekit_api_key="APIabc123", livekit_api_secret="s" * 40)


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _decode(token: str):
    header_b64, payload_b64, sig_b64 = token.split(".")
    header = json.loads(_b64url_decode(header_b64))
    payload = json.loads(_b64url_decode(payload_b64))
    return header, payload, header_b64, payload_b64, sig_b64


# --- is_configured ---


def test_is_configured():
    assert is_configured(_CFG)
    assert not is_configured(DEFAULT)  # no key/secret by default
    assert not is_configured(replace(_CFG, livekit_api_secret=""))


# --- token structure (what LiveKit Cloud validates) ---


def test_token_is_hs256_jwt_with_video_grant():
    token = access_token(identity="clinician", room="rmsai-call-1", config=_CFG,
                         name="On-call", ttl_seconds=600, now=1_000_000)
    header, payload, h_b64, p_b64, sig_b64 = _decode(token)

    assert header == {"alg": "HS256", "typ": "JWT"}
    assert payload["iss"] == "APIabc123"          # API key
    assert payload["sub"] == "clinician"          # identity
    assert payload["nbf"] == 1_000_000 and payload["exp"] == 1_000_600
    assert payload["name"] == "On-call"
    grant = payload["video"]
    assert grant["room"] == "rmsai-call-1" and grant["roomJoin"] is True
    assert grant["canPublish"] and grant["canSubscribe"]


def test_token_signature_verifies_with_secret():
    token = access_token(identity="agent", room="r", config=_CFG, now=1)
    _, _, h_b64, p_b64, sig_b64 = _decode(token)
    expected = hmac.new(b"s" * 40, f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
    assert _b64url_decode(sig_b64) == expected  # tamper-evident, correctly signed


def test_token_requires_credentials():
    with pytest.raises(ValueError):
        access_token(identity="x", room="r", config=DEFAULT)


# --- caller factory ---


def test_get_caller_simulated_default():
    from voice.outbound import SimulatedCaller

    assert isinstance(get_caller("simulated"), SimulatedCaller)


def test_get_caller_livekit():
    assert isinstance(get_caller("livekit", _CFG), LiveKitCaller)


def test_livekit_caller_without_config_returns_invalid():
    # not configured -> fails fast as INVALID (no SDK import, no network)
    assert LiveKitCaller(DEFAULT).place_call("+15551234567") == CallOutcome.INVALID
