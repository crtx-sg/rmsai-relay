"""Provision the telephony server's SIP objects for phone calls bridged in by Twilio Programmable Voice.

Target is `config.telephony()`, i.e. the LiveKit Cloud project in split mode. Twilio's trial blocks
Elastic SIP Trunking, so calls reach LiveKit the other way round: Twilio places or receives the
PSTN leg itself and its TwiML `<Dial><Sip>` bridges the answered call **into** LiveKit as an inbound
SIP call. The SIP user part names the room:

    outbound  relay → Twilio Calls API → your mobile answers → TwiML
              <Dial><Sip>sip:rmsai-outbound-<event_id>@<LIVEKIT_SIP_URI></Sip>   (P3b)
    inbound   your mobile → Twilio number → TwiML Bin (printed by `cli.sip_setup --twiml`)
              <Dial><Sip>sip:rmsai-call-{{CallSid}}@<LIVEKIT_SIP_URI></Sip>
    both      → inbound trunk (digest auth + caller allow-list) → callee dispatch rule
              → room named exactly by the user part → agent `sip_agent_name`

So one inbound trunk and one **callee** rule serve both directions. The outbound room is exactly the
one the relay staged the alert for (`randomize=False`); the inbound one is unique per call via
Twilio's `{{CallSid}}` template.

The rule dispatches the agent. In bridge mode the relay must therefore NOT also dispatch it
explicitly, or two agents join the room (see P3b).

Security: the trunk only accepts INVITEs carrying `sip_inbound_username/password`, and only from the
allowed callers (`inbound_allowed_numbers` for call-ins, plus `OUTBOUND_FROM`, the caller ID on
bridged outbound calls). The worker's PIN gate applies on top.

OPTIONAL paid path: when `TWILIO_SIP_*` is set (Elastic SIP Trunking), an outbound trunk
(`rmsai-outbound-twilio`) is also created, so upgrading later needs no code change.

**SignalWire** (`TELEPHONY_CARRIER=signalwire`) is a native SIP trunk both ways, per LiveKit's and
SignalWire's documented integration:

    outbound  LiveKit outbound trunk (TLS, digest creds) → SignalWire Domain App
              (`SIGNALWIRE_SIP_DOMAIN`) → SWML `connect` → PSTN       (swml_outbound)
    inbound   your SignalWire number → SWML `connect` to sip:<number>@<LIVEKIT_SIP_URI>
              (swml_inbound) → inbound trunk (matched on the number, caller allow-list)
              → individual dispatch rule → `rmsai-call-…` room → `sip_agent_name`

Outbound needs no bridge: the existing `LiveKitCaller` dial path uses the outbound trunk. Its id is
`LIVEKIT_SIP_TRUNK_ID`. The SignalWire objects have their own names, so both carriers' objects can
coexist in one project.

Idempotent by fixed name: an existing object is replaced in place, a missing one is created.
`plan()` is pure (plain dicts); `apply()` takes any object shaped like the SDK's `SipService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from common.config import DEFAULT, Config
from voice.outbound import is_valid_number, mask_number

CARRIERS = ("twilio", "signalwire")
OUTBOUND_TRUNK_NAME = "rmsai-outbound-twilio"
INBOUND_TRUNK_NAME = "rmsai-inbound-twilio"
DISPATCH_RULE_NAME = "rmsai-inbound-dispatch"
SW_OUTBOUND_TRUNK_NAME = "rmsai-outbound-signalwire"
SW_INBOUND_TRUNK_NAME = "rmsai-inbound-signalwire"
SW_DISPATCH_RULE_NAME = "rmsai-inbound-dispatch-signalwire"


@dataclass
class SipPlan:
    outbound: dict | None = None  # only on the paid Elastic-SIP path
    inbound: dict | None = None
    dispatch: dict | None = None  # `trunk_ids` is filled in by `apply` once the inbound id is known
    problems: list[str] = field(default_factory=list)  # blocking: nothing is applied
    notes: list[str] = field(default_factory=list)     # informational


@dataclass
class SipResult:
    outbound_trunk_id: str | None = None
    inbound_trunk_id: str | None = None
    dispatch_rule_id: str | None = None
    actions: list[str] = field(default_factory=list)


def _e164(number: str) -> bool:
    """Strict E.164 for trunk config: the shared `is_valid_number` allows a missing `+`, which is
    fine for dialling but not for allow-lists, which match the carrier's exact form."""
    return number.startswith("+") and is_valid_number(number)


def _host(uri: str) -> str:
    """`sip:abc.sip.livekit.cloud` / `abc.sip.livekit.cloud` → `abc.sip.livekit.cloud`."""
    host = uri.strip()
    return host[4:] if host.lower().startswith("sip:") else host


def sip_target(room: str, config: Config = DEFAULT) -> str:
    """The SIP URI Twilio dials to land a call in `room` on the telephony server."""
    return f"sip:{room}@{_host(config.livekit_sip_uri)}"


def dial_sip_twiml(room: str, config: Config = DEFAULT, *, caller_id: str | None = None) -> str:
    """TwiML that bridges the answered Twilio call into `room` (digest auth to the inbound trunk).

    `room` may carry a Twilio template (`{{CallSid}}`) when used in a TwiML Bin.
    """
    cid = f" callerId={quoteattr(caller_id)}" if caller_id else ""
    return (
        f"<Response><Dial{cid}>"
        f"<Sip username={quoteattr(config.sip_inbound_username)} "
        f"password={quoteattr(config.sip_inbound_password)}>{sip_target(room, config)}</Sip>"
        f"</Dial></Response>"
    )


def inbound_twiml_bin(config: Config = DEFAULT) -> str:
    """The TwiML Bin for the Twilio number's "A call comes in" handler: one room per call."""
    return dial_sip_twiml(f"{config.call_room_prefix}{{{{CallSid}}}}", config)


def swml_outbound(config: Config = DEFAULT) -> str:
    """SWML for the SignalWire Domain App: dial the PSTN number LiveKit asked for, from our number.

    Verbatim shape of SignalWire's LiveKit guide; `answer_on_bridge` keeps LiveKit's call ringing
    until the callee actually answers, so `wait_until_answered` means answered.
    """
    return (
        "version: 1.0.0\n"
        "sections:\n"
        "  main:\n"
        "    - connect:\n"
        "        answer_on_bridge: true\n"
        f'        from: "{config.outbound_from}"\n'
        "        to: \"%{call.to.replace(/^sip:/i, '').replace(/@.*/, '')}\"\n"
    )


def swml_inbound(config: Config = DEFAULT) -> str:
    """SWML for the SignalWire number: hand every inbound call to the telephony LiveKit."""
    return (
        "version: 1.0.0\n"
        "sections:\n"
        "  main:\n"
        "    - connect:\n"
        f'        to: "sip:%{{call.to}}@{_host(config.livekit_sip_uri)};transport=tcp"\n'
    )


def plan(config: Config = DEFAULT) -> SipPlan:
    """What should exist on the telephony server, derived from config alone (no network)."""
    if config.telephony_carrier not in CARRIERS:
        return SipPlan(problems=[f"TELEPHONY_CARRIER={config.telephony_carrier!r}: expected one of "
                                 f"{', '.join(CARRIERS)}"])
    if config.telephony_carrier == "signalwire":
        return _plan_signalwire(config)
    return _plan_twilio(config)


def _plan_signalwire(config: Config) -> SipPlan:
    p = SipPlan()
    tel = config.telephony()
    if not config.outbound_from:
        p.problems.append("OUTBOUND_FROM is empty: set it to your SignalWire number (E.164). It is "
                          "the trunk's number and the caller ID on every call")
    elif not _e164(config.outbound_from):
        p.problems.append(f"OUTBOUND_FROM {mask_number(config.outbound_from)} is not E.164 (+<digits>)")
    for key, val in (("SIGNALWIRE_SIP_DOMAIN", config.signalwire_sip_domain),
                     ("SIGNALWIRE_SIP_USERNAME", config.signalwire_sip_username),
                     ("SIGNALWIRE_SIP_PASSWORD", config.signalwire_sip_password)):
        if not val:
            p.problems.append(f"{key} is empty (SignalWire → the outbound SWML script's SIP address "
                              "on a Domain App, and its SIP credentials)")
    bad = [n for n in config.inbound_allowed_numbers if not _e164(n)]
    if bad:
        p.problems.append(f"SIP_INBOUND_ALLOWED_NUMBERS / OUTBOUND_CALL_NUMBER not E.164: "
                          f"{', '.join(mask_number(n) for n in bad)}")

    numbers = [config.outbound_from] if config.outbound_from else []
    p.outbound = {
        "name": SW_OUTBOUND_TRUNK_NAME,
        "address": _host(config.signalwire_sip_domain),
        "numbers": numbers,
        "auth_username": config.signalwire_sip_username,
        "auth_password": config.signalwire_sip_password,
        "transport": "tls",
    }
    allowed = list(config.inbound_allowed_numbers)
    if not allowed:
        p.notes.append("inbound skipped: no SIP_INBOUND_ALLOWED_NUMBERS and no OUTBOUND_CALL_NUMBER, "
                       "and an unrestricted inbound number is never created")
        return p
    p.inbound = {
        "name": SW_INBOUND_TRUNK_NAME,
        "numbers": numbers,          # the called number SignalWire's SWML forwards (call.to)
        "allowed_numbers": allowed,  # who may call in
        "max_call_duration_s": config.sip_max_call_duration_s,
    }
    p.dispatch = {
        "name": SW_DISPATCH_RULE_NAME,
        "kind": "individual",  # one room per call, <prefix>…
        "room_prefix": config.call_room_prefix,
        "agent_name": tel.livekit_agent_name,
        "hide_phone_number": True,
    }
    return p


def _plan_twilio(config: Config) -> SipPlan:
    p = SipPlan()
    tel = config.telephony()

    if not config.outbound_from:
        p.problems.append("OUTBOUND_FROM is empty: set it to your Twilio number (E.164). It is the "
                          "caller ID on bridged calls and must be an allowed caller on the trunk")
    elif not _e164(config.outbound_from):
        p.problems.append(f"OUTBOUND_FROM {mask_number(config.outbound_from)} is not E.164 (+<digits>)")
    if not config.livekit_sip_uri:
        p.problems.append("LIVEKIT_SIP_URI is empty (LiveKit Cloud → Settings → SIP URI, "
                          "e.g. abc123.sip.livekit.cloud)")
    for key, val in (("LIVEKIT_SIP_INBOUND_USERNAME", config.sip_inbound_username),
                     ("LIVEKIT_SIP_INBOUND_PASSWORD", config.sip_inbound_password)):
        if not val:
            p.problems.append(f"{key} is empty (choose one; Twilio's TwiML presents it to LiveKit)")
    bad = [n for n in config.inbound_allowed_numbers if not _e164(n)]
    if bad:
        p.problems.append(f"SIP_INBOUND_ALLOWED_NUMBERS / OUTBOUND_CALL_NUMBER not E.164: "
                          f"{', '.join(mask_number(n) for n in bad)}")

    # Paid path only: all three TWILIO_SIP_* or none.
    elastic = (config.twilio_sip_termination_uri, config.twilio_sip_username,
               config.twilio_sip_password)
    if any(elastic) and not all(elastic):
        p.problems.append("TWILIO_SIP_TERMINATION_URI/USERNAME/PASSWORD: set all three (Elastic "
                          "SIP Trunking, paid) or none (Programmable Voice bridge)")
    numbers = [config.outbound_from] if config.outbound_from else []
    if all(elastic):
        p.outbound = {
            "name": OUTBOUND_TRUNK_NAME,
            "address": _host(config.twilio_sip_termination_uri),
            "numbers": numbers,
            "auth_username": config.twilio_sip_username,
            "auth_password": config.twilio_sip_password,
        }
    else:
        p.notes.append("no outbound trunk: calls are placed by the Twilio Calls API and bridged in")

    allowed = list(dict.fromkeys([*config.inbound_allowed_numbers, *numbers]))
    if not config.inbound_allowed_numbers:
        p.notes.append("no call-in number allowed (OUTBOUND_CALL_NUMBER / SIP_INBOUND_ALLOWED_NUMBERS "
                       "empty): only bridged outbound calls will be accepted")
    p.inbound = {
        "name": INBOUND_TRUNK_NAME,
        "allowed_numbers": allowed,
        "auth_username": config.sip_inbound_username,
        "auth_password": config.sip_inbound_password,
        "max_call_duration_s": config.sip_max_call_duration_s,
    }
    p.dispatch = {
        "name": DISPATCH_RULE_NAME,
        "kind": "callee",  # room = user part of the dialled SIP URI, exactly
        "room_prefix": "",
        "randomize": False,
        "agent_name": tel.livekit_agent_name,
        "hide_phone_number": True,
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

    from livekit.protocol.sip import SIPTransport  # noqa: PLC0415

    info = SIPOutboundTrunkInfo(
        name=spec["name"], address=spec["address"], numbers=spec["numbers"],
        auth_username=spec["auth_username"], auth_password=spec["auth_password"],
    )
    if spec.get("transport"):
        info.transport = {"udp": SIPTransport.SIP_TRANSPORT_UDP, "tcp": SIPTransport.SIP_TRANSPORT_TCP,
                          "tls": SIPTransport.SIP_TRANSPORT_TLS}[spec["transport"]]
    return info


def _inbound_info(spec: dict):
    from google.protobuf.duration_pb2 import Duration  # noqa: PLC0415
    from livekit.protocol.sip import SIPInboundTrunkInfo  # noqa: PLC0415

    info = SIPInboundTrunkInfo(
        name=spec["name"], numbers=spec.get("numbers", []), allowed_numbers=spec["allowed_numbers"],
        auth_username=spec.get("auth_username", ""), auth_password=spec.get("auth_password", ""),
    )
    if spec.get("max_call_duration_s"):
        info.max_call_duration.CopyFrom(Duration(seconds=int(spec["max_call_duration_s"])))
    return info


def _dispatch_info(spec: dict, inbound_trunk_id: str):
    from livekit.protocol.agent_dispatch import RoomAgentDispatch  # noqa: PLC0415
    from livekit.protocol.room import RoomConfiguration  # noqa: PLC0415
    from livekit.protocol.sip import (  # noqa: PLC0415
        SIPDispatchRule,
        SIPDispatchRuleCallee,
        SIPDispatchRuleIndividual,
        SIPDispatchRuleInfo,
    )

    if spec["kind"] == "callee":
        rule = SIPDispatchRule(dispatch_rule_callee=SIPDispatchRuleCallee(
            room_prefix=spec["room_prefix"], randomize=spec["randomize"]))
    else:
        rule = SIPDispatchRule(dispatch_rule_individual=SIPDispatchRuleIndividual(
            room_prefix=spec["room_prefix"]))
    return SIPDispatchRuleInfo(
        name=spec["name"],
        trunk_ids=[inbound_trunk_id],
        hide_phone_number=spec["hide_phone_number"],
        rule=rule,
        room_config=RoomConfiguration(agents=[RoomAgentDispatch(agent_name=spec["agent_name"])]),
    )


# ------------------------------------------------------------------------------------ apply ---

async def _upsert(label, existing, name, info, create, update, id_attr, res) -> str:
    if name in existing:
        obj = await update(existing[name], info)
        verb = "updated"
    else:
        obj = await create(info)
        verb = "created"
    oid = getattr(obj, id_attr)
    res.actions.append(f"{verb} {label} {oid}")
    return oid


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
    if p.outbound is not None:
        existing = {t.name: t.sip_trunk_id
                    for t in (await sip.list_outbound_trunk(ListSIPOutboundTrunkRequest())).items}
        res.outbound_trunk_id = await _upsert(
            "outbound trunk", existing, p.outbound["name"], _outbound_info(p.outbound),
            lambda i: sip.create_outbound_trunk(CreateSIPOutboundTrunkRequest(trunk=i)),
            sip.update_outbound_trunk, "sip_trunk_id", res)

    if p.inbound is None:
        return res
    existing = {t.name: t.sip_trunk_id
                for t in (await sip.list_inbound_trunk(ListSIPInboundTrunkRequest())).items}
    res.inbound_trunk_id = await _upsert(
        "inbound trunk", existing, p.inbound["name"], _inbound_info(p.inbound),
        lambda i: sip.create_inbound_trunk(CreateSIPInboundTrunkRequest(trunk=i)),
        sip.update_inbound_trunk, "sip_trunk_id", res)

    existing = {r.name: r.sip_dispatch_rule_id
                for r in (await sip.list_dispatch_rule(ListSIPDispatchRuleRequest())).items}
    res.dispatch_rule_id = await _upsert(
        "dispatch rule", existing, p.dispatch["name"],
        _dispatch_info(p.dispatch, res.inbound_trunk_id),
        lambda i: sip.create_dispatch_rule(CreateSIPDispatchRuleRequest(dispatch_rule=i)),
        sip.update_dispatch_rule, "sip_dispatch_rule_id", res)
    return res
