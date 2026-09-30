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
    plan,
    redacted,
)

pytest.importorskip("livekit.protocol.sip")

_CFG = Config(
    livekit_url="ws://localhost:7880", livekit_api_key="devkey", livekit_api_secret="local",
    livekit_sip_url="wss://voiceapptest.livekit.cloud", livekit_sip_api_key="APIcloud",
    livekit_sip_api_secret="cloud-secret",
    outbound_from="+15550001000", outbound_call_number="+15550002000",
    twilio_sip_termination_uri="sip:rmsai.pstn.twilio.com",
    twilio_sip_username="relay", twilio_sip_password="twilio-pw",
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

def test_plan_outbound_trunk_dials_twilio_with_our_number():
    p = plan(_CFG)
    assert not p.problems
    assert p.outbound == {"name": OUTBOUND_TRUNK_NAME, "address": "rmsai.pstn.twilio.com",
                          "numbers": ["+15550001000"], "auth_username": "relay",
                          "auth_password": "twilio-pw"}


def test_plan_inbound_is_restricted_and_dispatch_names_the_phone_agent():
    p = plan(_CFG)
    assert p.inbound["allowed_numbers"] == ["+15550002000"]  # falls back to the on-call number
    assert p.inbound["numbers"] == ["+15550001000"]
    assert p.dispatch["agent_name"] == "rmsai-agent-phone"  # explicit dispatch: never an empty room
    assert p.dispatch["room_prefix"] == "rmsai-call-"


def test_plan_skips_inbound_when_nobody_is_allowed():
    p = plan(replace(_CFG, outbound_call_number="", sip_inbound_allowed_numbers=()))
    assert p.inbound is None and p.dispatch is None and not p.problems
    assert any("never created" in n for n in p.notes)


@pytest.mark.parametrize("field,value,expect", [
    ("outbound_from", "", "OUTBOUND_FROM is empty"),
    ("outbound_from", "5550001000", "not E.164"),
    ("twilio_sip_termination_uri", "", "TWILIO_SIP_TERMINATION_URI"),
    ("twilio_sip_password", "", "TWILIO_SIP_PASSWORD"),
    ("sip_inbound_allowed_numbers", ("12345",), "non-E.164"),
])
def test_plan_problems_block(field, value, expect):
    p = plan(replace(_CFG, **{field: value}))
    assert any(expect in prob for prob in p.problems)
    with pytest.raises(ValueError, match="nothing applied"):
        asyncio.run(apply(p, FakeSip()))


def test_redacted_masks_secret_and_numbers():
    text = json.dumps(redacted(plan(_CFG)))
    assert "twilio-pw" not in text and "+15550002000" not in text and "+15550001000" not in text
    assert "***" in text and "2000" in text  # masked, but still tellable apart


# ----------------------------------------------------------------------------------- apply ---

def test_apply_creates_all_three_and_links_rule_to_inbound_trunk():
    sip = FakeSip()
    res = asyncio.run(apply(plan(_CFG), sip))
    assert sip.calls == ["create_outbound", "create_inbound", "create_rule"]
    rule = sip.rules[res.dispatch_rule_id]
    assert list(rule.trunk_ids) == [res.inbound_trunk_id]
    assert rule.room_config.agents[0].agent_name == "rmsai-agent-phone"
    assert rule.rule.dispatch_rule_individual.room_prefix == "rmsai-call-"
    assert rule.hide_phone_number is True
    inbound = sip.inbound[res.inbound_trunk_id]
    assert list(inbound.allowed_numbers) == ["+15550002000"]
    assert inbound.max_call_duration.seconds == 600
    assert sip.outbound[res.outbound_trunk_id].address == "rmsai.pstn.twilio.com"


def test_apply_is_idempotent_updates_in_place():
    sip = FakeSip()
    first = asyncio.run(apply(plan(_CFG), sip))
    sip.calls.clear()
    changed = replace(_CFG, sip_inbound_allowed_numbers=("+15550003000",))
    second = asyncio.run(apply(plan(changed), sip))
    assert sip.calls == ["update_outbound", "update_inbound", "update_rule"]
    assert (second.outbound_trunk_id, second.inbound_trunk_id, second.dispatch_rule_id) == (
        first.outbound_trunk_id, first.inbound_trunk_id, first.dispatch_rule_id)
    assert len(sip.outbound) == len(sip.inbound) == len(sip.rules) == 1  # no duplicates
    assert list(sip.inbound[second.inbound_trunk_id].allowed_numbers) == ["+15550003000"]


def test_apply_outbound_only_when_inbound_skipped():
    sip = FakeSip()
    res = asyncio.run(apply(plan(replace(_CFG, outbound_call_number="")), sip))
    assert sip.calls == ["create_outbound"] and res.inbound_trunk_id is None


def test_names_are_stable_constants():
    # renaming these would orphan objects created by earlier runs (idempotency is by name)
    assert (OUTBOUND_TRUNK_NAME, INBOUND_TRUNK_NAME, DISPATCH_RULE_NAME) == (
        "rmsai-outbound-twilio", "rmsai-inbound-twilio", "rmsai-inbound-dispatch")


# ------------------------------------------------------------------------------------- CLI ---

def test_cli_dry_run_touches_nothing_and_masks(capsys):
    assert main(["--dry-run"], config=_CFG) == 0
    out = capsys.readouterr().out
    assert "wss://voiceapptest.livekit.cloud" in out and "rmsai-agent-phone" in out
    assert "dry run: nothing changed" in out and "Origination" in out
    assert "twilio-pw" not in out and "+15550002000" not in out


def test_cli_refuses_problems(capsys):
    assert main(["--dry-run"], config=replace(_CFG, outbound_from="")) == 2
    assert "PROBLEM" in capsys.readouterr().err


def test_cli_partial_telephony_config(capsys):
    assert main(["--dry-run"], config=replace(_CFG, livekit_sip_api_secret="")) == 2
    assert "LIVEKIT_SIP_API_SECRET" in capsys.readouterr().err
