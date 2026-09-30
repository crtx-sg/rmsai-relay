"""Provision the telephony server's SIP objects: outbound trunk, inbound trunk, dispatch rule.

Target is `config.telephony()`, i.e. the LiveKit Cloud project in split mode. The carrier is a
Twilio Elastic SIP Trunk:

    outbound  LiveKit ──INVITE──▶ <TWILIO_SIP_TERMINATION_URI> (credential auth) ──▶ PSTN
    inbound   PSTN ──▶ Twilio number ──▶ trunk origination URI = the project's SIP URI ──▶ LiveKit
              ──▶ dispatch rule: one `<call_room_prefix>…` room per call + agent `sip_agent_name`

The dispatch rule **names the agent**. Workers use explicit dispatch, so a rule without it drops the
caller into an empty room, which is the inbound gap in the old `sip-inbound.example.yaml`.

Inbound is **restricted**: the inbound trunk only accepts `config.inbound_allowed_numbers`, and
with none configured no inbound trunk or rule is created (never an open number). The PIN gate in
the worker still applies on top.

Idempotent by name: each object has a fixed `name`. An existing one is replaced in place (same id),
a missing one is created, so re-running converges instead of piling up duplicates.

`plan()` is pure (plain dicts, asserted offline); `apply()` takes any object shaped like the SDK's
`SipService`, so the create-vs-update logic is tested with a fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.config import DEFAULT, Config
from voice.outbound import is_valid_number, mask_number

OUTBOUND_TRUNK_NAME = "rmsai-outbound-twilio"
INBOUND_TRUNK_NAME = "rmsai-inbound-twilio"
DISPATCH_RULE_NAME = "rmsai-inbound-dispatch"


@dataclass
class SipPlan:
    outbound: dict | None = None
    inbound: dict | None = None
    dispatch: dict | None = None  # `trunk_ids` is filled in by `apply` once the inbound id is known
    problems: list[str] = field(default_factory=list)  # blocking: nothing is applied
    notes: list[str] = field(default_factory=list)     # informational (e.g. inbound skipped)


@dataclass
class SipResult:
    outbound_trunk_id: str | None = None
    inbound_trunk_id: str | None = None
    dispatch_rule_id: str | None = None
    actions: list[str] = field(default_factory=list)  # "created outbound trunk …" / "updated …"


def _e164(number: str) -> bool:
    """Strict E.164 for trunk config: the shared `is_valid_number` allows a missing `+`, which is
    fine for dialling but not for trunk numbers and allow-lists, which match the carrier's exact form."""
    return number.startswith("+") and is_valid_number(number)


def _host(uri: str) -> str:
    """`sip:rmsai.pstn.twilio.com` / `rmsai.pstn.twilio.com` → `rmsai.pstn.twilio.com`."""
    host = uri.strip()
    return host[4:] if host.lower().startswith("sip:") else host


def plan(config: Config = DEFAULT) -> SipPlan:
    """What should exist on the telephony server, derived from config alone (no network)."""
    p = SipPlan()
    tel = config.telephony()

    if not config.outbound_from:
        p.problems.append("OUTBOUND_FROM is empty: set it to your Twilio number (E.164). It is the "
                          "trunk's number and the caller ID on every call")
    elif not _e164(config.outbound_from):
        p.problems.append(f"OUTBOUND_FROM {mask_number(config.outbound_from)} is not E.164 (+<digits>)")
    for key, val in (("TWILIO_SIP_TERMINATION_URI", config.twilio_sip_termination_uri),
                     ("TWILIO_SIP_USERNAME", config.twilio_sip_username),
                     ("TWILIO_SIP_PASSWORD", config.twilio_sip_password)):
        if not val:
            p.problems.append(f"{key} is empty (Twilio console → Elastic SIP Trunking → your trunk "
                              "→ Termination)")
    bad = [n for n in config.inbound_allowed_numbers if not _e164(n)]
    if bad:
        p.problems.append(f"SIP_INBOUND_ALLOWED_NUMBERS has non-E.164 entries: "
                          f"{', '.join(mask_number(n) for n in bad)}")

    numbers = [config.outbound_from] if config.outbound_from else []
    p.outbound = {
        "name": OUTBOUND_TRUNK_NAME,
        "address": _host(config.twilio_sip_termination_uri),
        "numbers": numbers,
        "auth_username": config.twilio_sip_username,
        "auth_password": config.twilio_sip_password,
    }

    allowed = list(config.inbound_allowed_numbers)
    if not allowed:
        p.notes.append("inbound skipped: no SIP_INBOUND_ALLOWED_NUMBERS and no OUTBOUND_CALL_NUMBER, "
                       "and an unrestricted inbound number is never created")
        return p
    p.inbound = {
        "name": INBOUND_TRUNK_NAME,
        "numbers": numbers,
        "allowed_numbers": allowed,
        "max_call_duration_s": config.sip_max_call_duration_s,
    }
    p.dispatch = {
        "name": DISPATCH_RULE_NAME,
        "room_prefix": config.call_room_prefix,
        "agent_name": tel.livekit_agent_name,
        "hide_phone_number": True,  # keep the caller's number out of participant identity/attributes
    }
    return p


def redacted(p: SipPlan) -> dict:
    """The plan with secrets and phone numbers masked, for printing (`--dry-run`, logs)."""
    def mask(d: dict | None) -> dict | None:
        if d is None:
            return None
        out = dict(d)
        if out.get("auth_password"):
            out["auth_password"] = "***"
        for key in ("numbers", "allowed_numbers"):
            if key in out:
                out[key] = [mask_number(n) for n in out[key]]
        return out
    return {"outbound": mask(p.outbound), "inbound": mask(p.inbound), "dispatch": mask(p.dispatch)}


# ------------------------------------------------------------------ protobuf conversion (SDK) ---

def _outbound_info(spec: dict):
    from livekit.protocol.sip import SIPOutboundTrunkInfo  # noqa: PLC0415

    return SIPOutboundTrunkInfo(
        name=spec["name"], address=spec["address"], numbers=spec["numbers"],
        auth_username=spec["auth_username"], auth_password=spec["auth_password"],
    )


def _inbound_info(spec: dict):
    from google.protobuf.duration_pb2 import Duration  # noqa: PLC0415
    from livekit.protocol.sip import SIPInboundTrunkInfo  # noqa: PLC0415

    info = SIPInboundTrunkInfo(
        name=spec["name"], numbers=spec["numbers"], allowed_numbers=spec["allowed_numbers"],
    )
    if spec.get("max_call_duration_s"):
        info.max_call_duration.CopyFrom(Duration(seconds=int(spec["max_call_duration_s"])))
    return info


def _dispatch_info(spec: dict, inbound_trunk_id: str):
    from livekit.protocol.agent_dispatch import RoomAgentDispatch  # noqa: PLC0415
    from livekit.protocol.room import RoomConfiguration  # noqa: PLC0415
    from livekit.protocol.sip import (  # noqa: PLC0415
        SIPDispatchRule,
        SIPDispatchRuleIndividual,
        SIPDispatchRuleInfo,
    )

    return SIPDispatchRuleInfo(
        name=spec["name"],
        trunk_ids=[inbound_trunk_id],
        hide_phone_number=spec["hide_phone_number"],
        rule=SIPDispatchRule(
            dispatch_rule_individual=SIPDispatchRuleIndividual(room_prefix=spec["room_prefix"]),
        ),
        room_config=RoomConfiguration(agents=[RoomAgentDispatch(agent_name=spec["agent_name"])]),
    )


# ------------------------------------------------------------------------------------ apply ---

async def apply(p: SipPlan, sip) -> SipResult:
    """Create or replace each planned object on `sip` (the SDK's `SipService`, or a fake).

    Refuses a plan with problems before touching the server.
    """
    if p.problems:
        raise ValueError("SIP plan has problems; nothing applied:\n  - " + "\n  - ".join(p.problems))
    from livekit.protocol.sip import (  # noqa: PLC0415
        CreateSIPDispatchRuleRequest,
        CreateSIPInboundTrunkRequest,
        CreateSIPOutboundTrunkRequest,
        ListSIPDispatchRuleRequest,
        ListSIPInboundTrunkRequest,
        ListSIPOutboundTrunkRequest,
    )

    res = SipResult()

    existing = {t.name: t.sip_trunk_id
                for t in (await sip.list_outbound_trunk(ListSIPOutboundTrunkRequest())).items}
    info = _outbound_info(p.outbound)
    if p.outbound["name"] in existing:
        tid = existing[p.outbound["name"]]
        res.outbound_trunk_id = (await sip.update_outbound_trunk(tid, info)).sip_trunk_id
        res.actions.append(f"updated outbound trunk {res.outbound_trunk_id}")
    else:
        created = await sip.create_outbound_trunk(CreateSIPOutboundTrunkRequest(trunk=info))
        res.outbound_trunk_id = created.sip_trunk_id
        res.actions.append(f"created outbound trunk {res.outbound_trunk_id}")

    if p.inbound is None:
        return res

    existing = {t.name: t.sip_trunk_id
                for t in (await sip.list_inbound_trunk(ListSIPInboundTrunkRequest())).items}
    info = _inbound_info(p.inbound)
    if p.inbound["name"] in existing:
        tid = existing[p.inbound["name"]]
        res.inbound_trunk_id = (await sip.update_inbound_trunk(tid, info)).sip_trunk_id
        res.actions.append(f"updated inbound trunk {res.inbound_trunk_id}")
    else:
        created = await sip.create_inbound_trunk(CreateSIPInboundTrunkRequest(trunk=info))
        res.inbound_trunk_id = created.sip_trunk_id
        res.actions.append(f"created inbound trunk {res.inbound_trunk_id}")

    existing = {r.name: r.sip_dispatch_rule_id
                for r in (await sip.list_dispatch_rule(ListSIPDispatchRuleRequest())).items}
    info = _dispatch_info(p.dispatch, res.inbound_trunk_id)
    if p.dispatch["name"] in existing:
        rid = existing[p.dispatch["name"]]
        res.dispatch_rule_id = (await sip.update_dispatch_rule(rid, info)).sip_dispatch_rule_id
        res.actions.append(f"updated dispatch rule {res.dispatch_rule_id}")
    else:
        created = await sip.create_dispatch_rule(CreateSIPDispatchRuleRequest(dispatch_rule=info))
        res.dispatch_rule_id = created.sip_dispatch_rule_id
        res.actions.append(f"created dispatch rule {res.dispatch_rule_id}")
    return res
