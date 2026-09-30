"""cli.sip_setup: plan the telephony SIP objects from config; apply idempotently (fake SipService)."""

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
    dial_sip_twiml,
    inbound_twiml_bin,
    plan,
    redacted,
    sip_target,
)

pytest.importorskip("livekit.protocol.sip")

_CFG = Config(
    livekit_url="ws://localhost:7880", livekit_api_key="devkey", livekit_api_secret="local",
    livekit_sip_url="wss://voiceapptest.livekit.cloud", livekit_sip_api_key="APIcloud",
    livekit_sip_api_secret="cloud-secret", livekit_sip_uri="sip:abc123.sip.livekit.cloud",
    sip_inbound_username="rmsai", sip_inbound_password="sip-pw",
    outbound_from="+15550001000", outbound_call_number="+15550002000",
)
# the paid Elastic SIP Trunking path (optional)
_ELASTIC = replace(_CFG, twilio_sip_termination_uri="sip:rmsai.pstn.twilio.com",
                   twilio_sip_username="relay", twilio_sip_password="twilio-pw")


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

def test_bridge_plan_has_no_outbound_trunk():
    p = plan(_CFG)
    assert not p.problems and p.outbound is None
    assert any("Calls API" in n for n in p.notes)


def test_inbound_trunk_authenticates_and_allows_mobile_and_bridge_caller_id():
    p = plan(_CFG)
    assert p.inbound["auth_username"] == "rmsai" and p.inbound["auth_password"] == "sip-pw"
    # call-ins from the mobile, and bridged outbound calls presenting our Twilio number
    assert p.inbound["allowed_numbers"] == ["+15550002000", "+15550001000"]


def test_callee_rule_names_room_exactly_and_dispatches_phone_agent():
    d = plan(_CFG).dispatch
    assert (d["kind"], d["room_prefix"], d["randomize"]) == ("callee", "", False)
    assert d["agent_name"] == "rmsai-agent-phone"


def test_elastic_path_adds_outbound_trunk():
    p = plan(_ELASTIC)
    assert not p.problems
    assert p.outbound == {"name": OUTBOUND_TRUNK_NAME, "address": "rmsai.pstn.twilio.com",
                          "numbers": ["+15550001000"], "auth_username": "relay",
                          "auth_password": "twilio-pw"}


def test_no_call_in_number_still_accepts_bridged_outbound():
    p = plan(replace(_CFG, outbound_call_number=""))
    assert not p.problems and p.inbound["allowed_numbers"] == ["+15550001000"]
    assert any("only bridged outbound" in n for n in p.notes)


@pytest.mark.parametrize("field,value,expect", [
    ("outbound_from", "", "OUTBOUND_FROM is empty"),
    ("outbound_from", "5550001000", "not E.164"),
    ("livekit_sip_uri", "", "LIVEKIT_SIP_URI"),
    ("sip_inbound_username", "", "LIVEKIT_SIP_INBOUND_USERNAME"),
    ("sip_inbound_password", "", "LIVEKIT_SIP_INBOUND_PASSWORD"),
    ("sip_inbound_allowed_numbers", ("12345",), "not E.164"),
    ("twilio_sip_username", "relay", "set all three"),  # partial Elastic config
])
def test_plan_problems_block(field, value, expect):
    p = plan(replace(_CFG, **{field: value}))
    assert any(expect in prob for prob in p.problems)
    with pytest.raises(ValueError, match="nothing applied"):
        asyncio.run(apply(p, FakeSip()))


def test_redacted_masks_secrets_and_numbers():
    text = json.dumps(redacted(plan(_ELASTIC)))
    for secret in ("sip-pw", "twilio-pw", "+15550002000", "+15550001000"):
        assert secret not in text
    assert "***" in text and "2000" in text


# --------------------------------------------------------------------------------- TwiML ---

def test_sip_target_and_dial_twiml():
    assert sip_target("rmsai-outbound-e1", _CFG) == "sip:rmsai-outbound-e1@abc123.sip.livekit.cloud"
    xml = dial_sip_twiml("rmsai-outbound-e1", _CFG, caller_id="+15550001000")
    assert xml == ('<Response><Dial callerId="+15550001000"><Sip username="rmsai" password="sip-pw">'
                   "sip:rmsai-outbound-e1@abc123.sip.livekit.cloud</Sip></Dial></Response>")


def test_twiml_escapes_credentials():
    import xml.etree.ElementTree as ET

    tricky = """p"'<&>"""
    xml = dial_sip_twiml("r", replace(_CFG, sip_inbound_password=tricky))
    assert ET.fromstring(xml).find("Dial/Sip").get("password") == tricky  # round-trips intact


def test_inbound_twiml_bin_is_one_room_per_call():
    assert "sip:rmsai-call-{{CallSid}}@abc123.sip.livekit.cloud" in inbound_twiml_bin(_CFG)


# ----------------------------------------------------------------------------------- apply ---

def test_apply_bridge_creates_inbound_trunk_and_callee_rule():
    sip = FakeSip()
    res = asyncio.run(apply(plan(_CFG), sip))
    assert sip.calls == ["create_inbound", "create_rule"] and res.outbound_trunk_id is None
    rule = sip.rules[res.dispatch_rule_id]
    assert list(rule.trunk_ids) == [res.inbound_trunk_id]
    assert rule.rule.WhichOneof("rule") == "dispatch_rule_callee"
    assert rule.rule.dispatch_rule_callee.room_prefix == ""
    assert rule.rule.dispatch_rule_callee.randomize is False
    assert rule.room_config.agents[0].agent_name == "rmsai-agent-phone"
    inbound = sip.inbound[res.inbound_trunk_id]
    assert (inbound.auth_username, inbound.auth_password) == ("rmsai", "sip-pw")
    assert list(inbound.allowed_numbers) == ["+15550002000", "+15550001000"]
    assert inbound.max_call_duration.seconds == 600


def test_apply_elastic_also_creates_outbound_trunk():
    sip = FakeSip()
    res = asyncio.run(apply(plan(_ELASTIC), sip))
    assert sip.calls == ["create_outbound", "create_inbound", "create_rule"]
    assert sip.outbound[res.outbound_trunk_id].address == "rmsai.pstn.twilio.com"


def test_apply_is_idempotent_updates_in_place():
    sip = FakeSip()
    first = asyncio.run(apply(plan(_CFG), sip))
    sip.calls.clear()
    second = asyncio.run(apply(plan(replace(_CFG, sip_inbound_allowed_numbers=("+15550003000",))),
                               sip))
    assert sip.calls == ["update_inbound", "update_rule"]
    assert (second.inbound_trunk_id, second.dispatch_rule_id) == (
        first.inbound_trunk_id, first.dispatch_rule_id)
    assert len(sip.inbound) == len(sip.rules) == 1
    assert list(sip.inbound[second.inbound_trunk_id].allowed_numbers) == [
        "+15550003000", "+15550001000"]


def test_names_are_stable_constants():
    # renaming these would orphan objects created by earlier runs (idempotency is by name)
    assert (OUTBOUND_TRUNK_NAME, INBOUND_TRUNK_NAME, DISPATCH_RULE_NAME) == (
        "rmsai-outbound-twilio", "rmsai-inbound-twilio", "rmsai-inbound-dispatch")


# ------------------------------------------------------------------------------------- CLI ---

def test_cli_dry_run_touches_nothing_and_masks(capsys):
    assert main(["--dry-run"], config=_CFG) == 0
    out = capsys.readouterr().out
    assert "wss://voiceapptest.livekit.cloud" in out and "rmsai-agent-phone" in out
    assert "dry run: nothing changed" in out and "TwiML Bins" in out
    assert "sip-pw" not in out and "+15550002000" not in out


def test_cli_twiml_prints_the_bin(capsys):
    assert main(["--twiml"], config=_CFG) == 0
    out = capsys.readouterr().out.strip()
    assert out == inbound_twiml_bin(_CFG)


def test_cli_twiml_needs_sip_settings(capsys):
    assert main(["--twiml"], config=replace(_CFG, livekit_sip_uri="")) == 2
    assert "LIVEKIT_SIP_URI" in capsys.readouterr().err


def test_cli_refuses_problems(capsys):
    assert main(["--dry-run"], config=replace(_CFG, outbound_from="")) == 2
    assert "PROBLEM" in capsys.readouterr().err


def test_cli_partial_telephony_config(capsys):
    assert main(["--dry-run"], config=replace(_CFG, livekit_sip_api_secret="")) == 2
    assert "LIVEKIT_SIP_API_SECRET" in capsys.readouterr().err
