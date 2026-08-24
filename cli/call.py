"""Ring the predefined on-call number on demand (Phase A of the phone pipeline).

Not tied to an event: this is "get a clinician on the phone now". Each call gets its own room
(`rmsai-call-<id>`), the agent is dispatched into it *before* the dial so somebody is there when the
callee picks up, and the worker — finding no staged alert in that room — runs the PIN-gated Q&A
handler. So the callee hears the PIN prompt, authenticates, and can then ask grounded questions.

  # dry run against the simulated caller (no telephony, no LiveKit needed)
  uv run python -m cli.call

  # the real thing: dials OUTBOUND_CALL_NUMBER through LIVEKIT_SIP_TRUNK_ID
  uv run python -m cli.call --caller livekit

  # a different destination, and no agent (just prove the trunk rings the phone)
  uv run python -m cli.call --caller livekit --to +15551234567 --no-dispatch

Needs, for `--caller livekit`: `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`, an outbound
trunk in `LIVEKIT_SIP_TRUNK_ID`, a caller ID in `OUTBOUND_FROM` (most carriers reject a call with
no valid from-number), and the voice worker running so the dispatched agent has somewhere to land.
"""

from __future__ import annotations

import argparse

from common.audit import AuditLog
from common.config import DEFAULT
from voice.outbound import CallOutcome, get_caller, mask_number, place_predefined_call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", default=None,
                        help="Destination (default: OUTBOUND_CALL_NUMBER).")
    parser.add_argument("--caller", default="simulated", choices=["simulated", "livekit"],
                        help="simulated = no telephony (default); livekit = real SIP.")
    parser.add_argument("--call-id", default=None, help="Room suffix (default: random).")
    parser.add_argument("--no-dispatch", action="store_true",
                        help="Don't dispatch the agent — dial into an empty room (trunk test).")
    args = parser.parse_args(argv)

    config = DEFAULT
    number = args.to or config.outbound_call_number
    if not number:
        parser.error("no destination: set OUTBOUND_CALL_NUMBER in .env or pass --to")

    dispatcher = None
    if not args.no_dispatch:
        from voice.livekit_cloud import create_agent_dispatch, is_configured  # noqa: PLC0415

        if is_configured(config):
            dispatcher = lambda room: create_agent_dispatch(room, config=config)  # noqa: E731
        else:
            print("[call] LiveKit not configured; skipping agent dispatch (nobody will answer).")

    # The room name is decided inside place_predefined_call so the dispatch and the dial can never
    # disagree about it — the failure that leaves an agent waiting in a room the call never enters.
    room, outcome, attempts = place_predefined_call(
        config=config, caller=get_caller(args.caller, config), number=number,
        call_id=args.call_id, dispatcher=dispatcher, audit=AuditLog(config.audit_log_path),
    )

    print(f"[call] room={room} to={mask_number(number)} caller={args.caller} "
          f"outcome={outcome.value} attempts={attempts}")
    if outcome == CallOutcome.ANSWERED:
        print(f"[call] answered — the agent should now be talking in {room!r}. "
              f"Watch the worker console for '[worker] joined room {room!r}'.")
        return 0
    if outcome == CallOutcome.INVALID:
        print("[call] INVALID — the number failed validation, or LiveKit/the SIP trunk is not "
              "configured. Retrying would not help; fix the config and re-run.")
        return 2
    print(f"[call] {outcome.value} after {attempts} attempt(s) — the phone did not pick up.")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
