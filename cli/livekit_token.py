"""Mint a LiveKit join token — to drive the real WebRTC audio loop from a browser (no SIP/phone).

  uv run python -m cli.livekit_token --room rmsai-call-demo
  # then open https://agents-playground.livekit.io  -> "Manual" -> paste the URL + token,
  # allow the mic, and talk to the agent. (Or: lk room join --identity me <room>.)

Used both ways:
  * inbound  — join any room (e.g. rmsai-call-demo); the worker greets + PIN-gates, then answers
    KB-grounded questions.
  * outbound — join the rmsai-outbound-<event_id> room printed by `cli.consume --transport webrtc`;
    the worker reads that event's staged alert and speaks it after the PIN gate.

The agent worker (`cli.voice_worker dev`) must be running and auto-dispatching to new rooms.
"""

from __future__ import annotations

import argparse

from common.config import Config
from voice.livekit_cloud import access_token, is_configured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--room", required=True, help="Room to join (must match the worker's room).")
    parser.add_argument("--identity", default="clinician", help="Participant identity.")
    parser.add_argument("--name", default="clinician", help="Display name.")
    parser.add_argument("--ttl", type=int, default=3600, help="Token lifetime (seconds).")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if not is_configured(config):
        raise SystemExit(
            "LiveKit is not configured. Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET "
            "(LiveKit Cloud: wss://<project>.livekit.cloud) in your .env."
        )

    token = access_token(
        identity=args.identity, room=args.room, name=args.name,
        ttl_seconds=args.ttl, config=config,
    )
    print(f"URL:   {config.livekit_url}")
    print(f"Room:  {args.room}")
    print(f"Token: {token}")
    print("\nJoin: https://agents-playground.livekit.io -> Manual -> paste URL + Token, allow mic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
