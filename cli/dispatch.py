"""Dispatch the LiveKit agent worker into a room — re-wire in-app chat/voice without an app re-login.

A worker registered under a name (explicit dispatch) does NOT auto-join rooms; the gateway only
dispatches at `POST /session`. So after you **restart the worker**, rooms the app already joined are
left agent-less: chat/select reach an empty room (no logs, no replies, no speech). This re-dispatches
the agent into them. (The worker also does this automatically on startup — see
`LIVEKIT_REDISPATCH_ON_START` — this CLI is the on-demand equivalent.)

  uv run python -m cli.dispatch --room rmsai-inbox-h1     # one specific room
  uv run python -m cli.dispatch --all-inbox               # every live rmsai-inbox-* room lacking an agent

Idempotent: a room that already has an agent is left alone.
"""

from __future__ import annotations

import argparse

from common.config import Config
from voice.livekit_cloud import (
    create_agent_dispatch,
    is_configured,
    redispatch_existing_rooms,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--room", help="Dispatch the agent into this specific room.")
    target.add_argument("--all-inbox", action="store_true",
                        help="Dispatch into every live rmsai-inbox-* room that lacks an agent.")
    parser.add_argument("--include-empty", action="store_true",
                        help="With --all-inbox, also dispatch into rooms with no human participant "
                             "(default: skip empty/stale rooms).")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if not is_configured(config):
        raise SystemExit(
            "LiveKit is not configured. Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET "
            "(LiveKit Cloud: wss://<project>.livekit.cloud) in your .env."
        )

    if args.all_inbox:
        dispatched = redispatch_existing_rooms(config, require_human=not args.include_empty)
        if dispatched:
            for room in dispatched:
                print(f"[dispatch] agent {config.livekit_agent_name!r} requested for room {room}")
            print(f"[dispatch] {len(dispatched)} room(s) dispatched. "
                  "The agent joins in a few seconds (cold-start loads STT/TTS/orchestrator).")
        else:
            print("[dispatch] nothing to do — every live rmsai-inbox-* room already has an agent "
                  "(or none are live).")
        return 0

    try:
        created = create_agent_dispatch(args.room, config=config)
    except Exception as exc:  # noqa: BLE001 - surface a clear failure for the operator
        raise SystemExit(f"[dispatch] failed to dispatch into {args.room}: "
                         f"{type(exc).__name__}: {exc}")
    if created:
        print(f"[dispatch] agent {config.livekit_agent_name!r} requested for room {args.room}. "
              "It joins in a few seconds (cold-start loads STT/TTS/orchestrator).")
    else:
        print(f"[dispatch] agent already present in room {args.room} — nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
