"""cli.sip_setup: plan the SignalWire SIP objects from config; apply idempotently (fake SipService)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cli.sip_setup import main
from common.config import Config
from voice.sip_setup import (
    DISPATCH_RULE_NAME,
    INBOUND_TRUNK_NAME,
    OUTBOUND_TRUNK_NAME,
    apply,
    plan,
    redacted,
)

pytest.importorskip("livekit.protocol.sip")

_SW = Config(
    livekit_url="ws://localhost:7880", livekit_api_key="devkey", livekit_api_secret="local",
    livekit_sip_url="wss://voiceapptest.livekit.cloud", livekit_sip_api_key="APIcloud",
    livekit_sip_api_secret="cloud-secret", livekit_sip_uri="sip:abc123.sip.livekit.cloud",
    outbound_from="+15550001000", outbound_call_number="+15550002000",
    telephony_carrier="signalwire",
    signalwire_sip_domain="sip:rmsai.dapp.signalwire.com",
    signalwire_sip_username="lk-out", signalwire_sip_password="sw-pw",
)


class FakeSip:
    """In-memory stand-in for the SDK's SipService: records every call, assigns ids on create."""

    def __init__(self):
        self.outbound, self.inbound, self.rules, self.calls = {}, {}, {}, []
        self._n = 0

    def _id(self, prefix):
        self._n += 1
        return f"{prefix}_{self._n}"

    async def list_outbound_trunk(self, _req):
        return SimpleNamespace(items=list(self.outbound.values()))

    async def list_inbound_trunk(self, _req):
        return SimpleNamespace(items=list(self.inbound.values()))

    async def list_dispatch_rule(self, _req):
        return SimpleNamespace(items=list(self.rules.values()))

    async def create_outbound_trunk(self, req):
        self.calls.append("create_outbound")
        req.trunk.sip_trunk_id = self._id("ST")
        self.outbound[req.trunk.sip_trunk_id] = req.trunk
        return req.trunk

    async def update_outbound_trunk(self, tid, info):
        self.calls.append("update_outbound")
        info.sip_trunk_id = tid
        self.outbound[tid] = info
        return info

    async def create_inbound_trunk(self, req):
        self.calls.append("create_inbound")
        req.trunk.sip_trunk_id = self._id("ST")
        self.inbound[req.trunk.sip_trunk_id] = req.trunk
        return req.trunk

    async def update_inbound_trunk(self, tid, info):
        self.calls.append("update_inbound")
        info.sip_trunk_id = tid
        self.inbound[tid] = info
        return info

    async def create_dispatch_rule(self, req):
        self.calls.append("create_rule")
        req.dispatch_rule.sip_dispatch_rule_id = self._id("SDR")
        self.rules[req.dispatch_rule.sip_dispatch_rule_id] = req.dispatch_rule
        return req.dispatch_rule

    async def update_dispatch_rule(self, rid, info):
        self.calls.append("update_rule")
        info.sip_dispatch_rule_id = rid
        self.rules[rid] = info
        return info


# ------------------------------------------------------------------------------------ plan ---

def test_signalwire_is_the_default_carrier():
    assert Config().telephony_carrier == "signalwire"


def test_redacted_masks_secrets_and_numbers():
    text = json.dumps(redacted(plan(_SW)))
    for secret in ("sw-pw", "+15550002000", "+15550001000"):
        assert secret not in text
    assert "***" in text and "2000" in text


def test_sw_allowed_callers_must_be_e164():
    p = plan(replace(_SW, sip_inbound_allowed_numbers=("12345",)))
    assert any("not E.164" in prob for prob in p.problems)
    with pytest.raises(ValueError, match="nothing applied"):
        asyncio.run(apply(p, FakeSip()))


def test_sw_apply_updates_the_allow_list_in_place():
    sip = FakeSip()
    first = asyncio.run(apply(plan(_SW), sip))
    sip.calls.clear()
    second = asyncio.run(apply(plan(replace(_SW, sip_inbound_allowed_numbers=("+15550003000",))),
                               sip))
    assert sip.calls == ["update_outbound", "update_inbound", "update_rule"]
    assert (second.inbound_trunk_id, second.dispatch_rule_id) == (
        first.inbound_trunk_id, first.dispatch_rule_id)
    assert list(sip.inbound[second.inbound_trunk_id].allowed_numbers) == ["+15550003000"]


def test_names_are_stable_constants():
    # renaming these would orphan objects created by earlier runs (idempotency is by name)
    assert (OUTBOUND_TRUNK_NAME, INBOUND_TRUNK_NAME, DISPATCH_RULE_NAME) == (
        "rmsai-outbound-signalwire", "rmsai-inbound-signalwire", "rmsai-inbound-dispatch-signalwire")


def test_cli_refuses_problems(capsys):
    assert main(["--dry-run"], config=replace(_SW, outbound_from="")) == 2
    assert "PROBLEM" in capsys.readouterr().err


def test_cli_partial_telephony_config(capsys):
    assert main(["--dry-run"], config=replace(_SW, livekit_sip_api_secret="")) == 2
    assert "LIVEKIT_SIP_API_SECRET" in capsys.readouterr().err


# ------------------------------------------------------------------------------ SignalWire ---

def test_sw_outbound_trunk_dials_the_domain_app_over_tls():
    p = plan(_SW)
    assert not p.problems
    assert p.outbound == {"name": OUTBOUND_TRUNK_NAME, "address": "rmsai.dapp.signalwire.com",
                          "numbers": ["+15550001000"], "auth_username": "lk-out",
                          "auth_password": "sw-pw", "transport": "tls"}


def test_sw_inbound_matches_our_number_and_restricts_callers():
    p = plan(_SW)
    assert p.inbound["numbers"] == ["+15550001000"]          # the number SWML forwards (call.to)
    assert p.inbound["allowed_numbers"] == ["+15550002000"]  # callers: the mobile only
    assert "auth_username" not in p.inbound                  # SWML connect presents no digest creds
    d = p.dispatch
    assert (d["kind"], d["room_prefix"], d["agent_name"]) == (
        "individual", "rmsai-call-", "rmsai-agent-phone")


@pytest.mark.parametrize("field,expect", [
    ("signalwire_sip_domain", "SIGNALWIRE_SIP_DOMAIN"),
    ("signalwire_sip_username", "SIGNALWIRE_SIP_USERNAME"),
    ("signalwire_sip_password", "SIGNALWIRE_SIP_PASSWORD"),
    ("outbound_from", "OUTBOUND_FROM"),
])
def test_sw_problems_block(field, expect):
    p = plan(replace(_SW, **{field: ""}))
    assert any(expect in prob for prob in p.problems)


def test_unknown_carrier_is_refused():
    p = plan(replace(_SW, telephony_carrier="vonage"))
    assert p.problems and "TELEPHONY_CARRIER" in p.problems[0]


def test_sw_no_call_in_number_keeps_outbound_only():
    p = plan(replace(_SW, outbound_call_number=""))
    assert not p.problems and p.outbound and p.inbound is None and p.dispatch is None
    sip = FakeSip()
    res = asyncio.run(apply(p, sip))
    assert sip.calls == ["create_outbound"] and res.outbound_trunk_id


def test_sw_apply_creates_trunks_and_individual_rule_idempotently():
    from livekit.protocol.sip import SIPTransport

    sip = FakeSip()
    first = asyncio.run(apply(plan(_SW), sip))
    assert sip.calls == ["create_outbound", "create_inbound", "create_rule"]
    out = sip.outbound[first.outbound_trunk_id]
    assert out.transport == SIPTransport.SIP_TRANSPORT_TLS and out.auth_password == "sw-pw"
    inbound = sip.inbound[first.inbound_trunk_id]
    assert list(inbound.numbers) == ["+15550001000"] and inbound.auth_username == ""
    rule = sip.rules[first.dispatch_rule_id]
    assert rule.rule.WhichOneof("rule") == "dispatch_rule_individual"
    assert rule.rule.dispatch_rule_individual.room_prefix == "rmsai-call-"
    assert rule.room_config.agents[0].agent_name == "rmsai-agent-phone"
    sip.calls.clear()
    second = asyncio.run(apply(plan(_SW), sip))
    assert sip.calls == ["update_outbound", "update_inbound", "update_rule"]
    assert second.outbound_trunk_id == first.outbound_trunk_id


def test_swml_scripts_are_valid_and_point_at_the_right_places():
    import yaml

    from voice.sip_setup import swml_inbound, swml_outbound

    out = yaml.safe_load(swml_outbound(_SW))["sections"]["main"][0]["connect"]
    assert out["from"] == "+15550001000" and out["answer_on_bridge"] is True
    assert out["to"] == "%{call.to.replace(/^sip:/i, '').replace(/@.*/, '')}"
    inb = yaml.safe_load(swml_inbound(_SW))["sections"]["main"][0]["connect"]
    assert inb["to"] == "sip:%{call.to}@abc123.sip.livekit.cloud;transport=tcp"


def test_swml_outbound_reads_the_destination_header_with_a_fixed_sip_user():
    import yaml

    from voice.sip_setup import swml_outbound

    out = yaml.safe_load(swml_outbound(replace(_SW, signalwire_sip_user="rmsai")))
    conn = out["sections"]["main"][0]["connect"]
    assert conn["from"] == "+15550001000" and conn["answer_on_bridge"] is True
    assert "x-rmsai-to" in conn["to"] and "call.headers" in conn["to"]


def test_cli_swml_prints_both_scripts(capsys):
    assert main(["--swml"], config=_SW) == 0
    out = capsys.readouterr().out
    assert "outbound" in out and "inbound" in out
    assert "answer_on_bridge: true" in out and "abc123.sip.livekit.cloud" in out


def test_cli_swml_needs_number_and_sip_uri(capsys):
    assert main(["--swml"], config=replace(_SW, livekit_sip_uri="")) == 2
    assert "LIVEKIT_SIP_URI" in capsys.readouterr().err


def test_cli_dry_run_signalwire(capsys):
    assert main(["--dry-run"], config=_SW) == 0
    out = capsys.readouterr().out
    assert "carrier=signalwire" in out and "SignalWire side" in out
    assert "sw-pw" not in out and "+15550002000" not in out
