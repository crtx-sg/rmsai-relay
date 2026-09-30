"""Provision the phone-call SIP objects on the telephony LiveKit (inbound trunk + callee rule).

  python -m cli.sip_setup --dry-run     # print the plan (secrets and numbers masked), touch nothing
  python -m cli.sip_setup               # create or update in place
  python -m cli.sip_setup --twiml       # print the TwiML Bin to paste into Twilio (contains the password)

Targets `config.telephony()`: the LiveKit Cloud project when `LIVEKIT_SIP_URL` is set, else
`LIVEKIT_URL`. Idempotent: re-running updates the same objects by name. Calls are bridged in by
Twilio Programmable Voice (`<Dial><Sip>`), which works on a trial account; see voice/sip_setup.py.
The Twilio-side steps are printed as a checklist; this CLI cannot set them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from common.config import DEFAULT, Config
from voice.sip_setup import SipPlan, apply, inbound_twiml_bin, plan, redacted


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
                        help="print the inbound TwiML Bin (unmasked: it carries the SIP password)")
    args = parser.parse_args(argv)

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
    print(f"[sip_setup] target {tel.livekit_url}  mode={mode}  agent={tel.livekit_agent_name}")

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
        print(_twilio_checklist(config))
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
    if res.outbound_trunk_id:  # paid Elastic-SIP path only
        print(f"\nSet in .env:\n  LIVEKIT_SIP_TRUNK_ID={res.outbound_trunk_id}")
    print(_twilio_checklist(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
