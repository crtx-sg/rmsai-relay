"""Provision the phone-call SIP objects on the telephony LiveKit (inbound trunk + callee rule).

  python -m cli.sip_setup --dry-run     # print the plan (secrets and numbers masked), touch nothing
  python -m cli.sip_setup               # create or update in place
  python -m cli.sip_setup --twiml       # Twilio: the TwiML Bin to paste (contains the password)
  python -m cli.sip_setup --swml        # SignalWire: the outbound + inbound SWML scripts to paste

Targets `config.telephony()`: the LiveKit Cloud project when `LIVEKIT_SIP_URL` is set, else
`LIVEKIT_URL`. Idempotent: re-running updates the same objects by name. Calls are bridged in by
Twilio Programmable Voice (`<Dial><Sip>`, works on a trial account), or over a native SignalWire SIP
trunk (`TELEPHONY_CARRIER=signalwire`); see voice/sip_setup.py. The carrier-side steps are printed
as a checklist; this CLI cannot set them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from common.config import DEFAULT, Config
from voice.sip_setup import (
    SipPlan,
    apply,
    inbound_twiml_bin,
    plan,
    redacted,
    swml_inbound,
    swml_outbound,
)


def _signalwire_checklist(config: Config) -> str:
    return (
        "SignalWire side (dashboard):\n"
        "  1. Phone Numbers: buy a number = OUTBOUND_FROM\n"
        "  2. Resources → Add New → SWML Script 'rmsai-outbound' = `python -m cli.sip_setup --swml` "
        "(outbound)\n"
        "     → Addresses & Phone Numbers → Add → SIP Address for it = SIGNALWIRE_SIP_DOMAIN "
        "(+ SIP credentials)\n"
        "  3. Resources → Add New → SWML Script 'rmsai-inbound' = the inbound script\n"
        "     → Phone Numbers → your number → Edit Settings → Assign Resource → rmsai-inbound\n"
        "  Test: `python -m cli.call --caller livekit` rings OUTBOUND_CALL_NUMBER; calling your "
        "SignalWire number reaches the agent's PIN prompt\n"
        "  (the phone worker must be running: `make phone-up`)"
    )


def _checklist(config: Config) -> str:
    if config.telephony_carrier == "signalwire":
        return _signalwire_checklist(config)
    return _twilio_checklist(config)


def _twilio_checklist(config: Config) -> str:
    return (
        "Twilio side (console):\n"
        "  1. Verified Caller IDs: your mobile (trial accounts only call/text verified numbers)\n"
        "  2. TwiML Bins → create 'rmsai-inbound' with the XML from `python -m cli.sip_setup --twiml`\n"
        "  3. Phone Numbers → your number (OUTBOUND_FROM) → Voice → 'A call comes in' = TwiML Bin "
        "rmsai-inbound\n"
        "  Test: call your Twilio number from your mobile → trial notice → PIN prompt from the agent\n"
        "  (the phone worker must be running: `make phone-up`)"
    )


async def _apply_live(p: SipPlan, config: Config):  # pragma: no cover - needs a live LiveKit
    from livekit import api  # noqa: PLC0415

    from voice.livekit_cloud import _http_url  # noqa: PLC0415

    lk = api.LiveKitAPI(url=_http_url(config.livekit_url), api_key=config.livekit_api_key,
                        api_secret=config.livekit_api_secret)
    try:
        return await apply(p, lk.sip)
    finally:
        await lk.aclose()


def main(argv: list[str] | None = None, *, config: Config = DEFAULT) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    parser.add_argument("--twiml", action="store_true",
                        help="Twilio: print the inbound TwiML Bin (unmasked: it carries the SIP password)")
    parser.add_argument("--swml", action="store_true",
                        help="SignalWire: print the outbound and inbound SWML scripts to paste")
    args = parser.parse_args(argv)

    if args.swml:
        missing = [k for k, v in (("OUTBOUND_FROM", config.outbound_from),
                                  ("LIVEKIT_SIP_URI", config.livekit_sip_uri)) if not v]
        if missing:
            print(f"[sip_setup] set {', '.join(missing)} first", file=sys.stderr)
            return 2
        print("# --- outbound: SWML script on the SignalWire Domain App (SIGNALWIRE_SIP_DOMAIN) ---")
        print(swml_outbound(config))
        print("# --- inbound: SWML script assigned to your SignalWire number ---")
        print(swml_inbound(config))
        return 0

    if args.twiml:
        missing = [k for k, v in (("LIVEKIT_SIP_URI", config.livekit_sip_uri),
                                  ("LIVEKIT_SIP_INBOUND_USERNAME", config.sip_inbound_username),
                                  ("LIVEKIT_SIP_INBOUND_PASSWORD", config.sip_inbound_password))
                   if not v]
        if missing:
            print(f"[sip_setup] set {', '.join(missing)} first", file=sys.stderr)
            return 2
        print(inbound_twiml_bin(config))
        return 0

    try:
        tel = config.telephony()
    except ValueError as exc:
        print(f"[sip_setup] {exc}", file=sys.stderr)
        return 2
    mode = "split (calls on a separate LiveKit)" if config.telephony_split else "single-server"
    print(f"[sip_setup] target {tel.livekit_url}  mode={mode}  carrier={config.telephony_carrier}  "
          f"agent={tel.livekit_agent_name}")

    p = plan(config)
    print(json.dumps(redacted(p), indent=2))
    for note in p.notes:
        print(f"[sip_setup] note: {note}")
    for problem in p.problems:
        print(f"[sip_setup] PROBLEM: {problem}", file=sys.stderr)
    if p.problems:
        return 2
    if args.dry_run:
        print("[sip_setup] dry run: nothing changed")
        if p.outbound is not None:
            print("[sip_setup] next: run again WITHOUT --dry-run to create these; it then prints the "
                  "LIVEKIT_SIP_TRUNK_ID=ST_… line for .env (no id exists until the trunk is created)")
        print(_checklist(config))
        return 0

    if not (tel.livekit_url and tel.livekit_api_key and tel.livekit_api_secret):
        print("[sip_setup] telephony LiveKit not configured (LIVEKIT_SIP_URL/KEY/SECRET)",
              file=sys.stderr)
        return 2
    try:
        res = asyncio.run(_apply_live(p, tel))
    except Exception as exc:  # noqa: BLE001 - surface the server's reason, never a traceback of secrets
        print(f"[sip_setup] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for action in res.actions:
        print(f"[sip_setup] {action}")
    if res.outbound_trunk_id:  # SignalWire, or Twilio's paid Elastic SIP path
        print(f"\nSet in .env:\n  LIVEKIT_SIP_TRUNK_ID={res.outbound_trunk_id}")
    print(_checklist(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
