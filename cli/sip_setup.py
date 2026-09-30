"""Provision the phone-call SIP objects on the telephony LiveKit (outbound/inbound trunk + dispatch).

  python -m cli.sip_setup --dry-run     # print the plan (secrets and numbers masked), touch nothing
  python -m cli.sip_setup               # create or update in place; prints LIVEKIT_SIP_TRUNK_ID

Targets `config.telephony()`: the LiveKit Cloud project when `LIVEKIT_SIP_URL` is set, else
`LIVEKIT_URL`. Idempotent: re-running updates the same objects by name. Twilio-side steps (the
Elastic SIP Trunk's termination credentials and its origination URI) are printed as a checklist;
this CLI cannot set them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from common.config import DEFAULT, Config
from voice.sip_setup import SipPlan, apply, plan, redacted


def _twilio_checklist(config: Config) -> str:
    return (
        "Twilio side (console → Elastic SIP Trunking → your trunk):\n"
        f"  Termination: SIP URI {config.twilio_sip_termination_uri or '<name>.pstn.twilio.com'}, "
        "credential list = TWILIO_SIP_USERNAME / TWILIO_SIP_PASSWORD\n"
        "  Origination: URI = this LiveKit project's SIP URI (LiveKit Cloud → Settings → SIP URI), "
        "e.g. sip:<id>.sip.livekit.cloud\n"
        "  Numbers:     attach OUTBOUND_FROM to the trunk"
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
    args = parser.parse_args(argv)

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
    print(f"\nSet in .env:\n  LIVEKIT_SIP_TRUNK_ID={res.outbound_trunk_id}")
    print(_twilio_checklist(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
