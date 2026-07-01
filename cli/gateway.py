"""Run the Phase 9 companion-app gateway (FastAPI via uvicorn).

  # serve the worklist app + endpoints (needs the `app` extra; LIVEKIT_* + HOSPITAL_ID for tokens)
  uv run python -m cli.gateway --host 0.0.0.0 --port 8080

Then open http://localhost:<port>/ , enter the PIN, and the app joins rmsai-inbox-<HOSPITAL_ID>
and renders the live worklist. Point the orchestrator (`cli.consume`) at the same HOSPITAL_ID with
DISPATCH_MODE including `app` so critical events are pushed into that room.
"""

from __future__ import annotations

import argparse

from common.config import Config
from live.gateway import create_app
from live.inbox import InboxPublisher, inbox_room
from voice.livekit_cloud import is_configured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    config = Config.from_env()
    print(f"[gateway] serving companion app on http://{args.host}:{args.port}/ "
          f"-> inbox room {inbox_room(config)}")

    # Graph driver backs the acknowledge round-trip; the publisher pushes the status change back to
    # the inbox (only when LiveKit is configured).
    from kb.graph.driver import GraphDriver  # noqa: PLC0415

    from live.artifact_tokens import ArtifactTokenStore  # noqa: PLC0415

    driver = GraphDriver.from_config(config)
    token_store = ArtifactTokenStore.from_config(config)  # shares Redis with the consumer's minter
    publisher = None
    if is_configured(config):
        publisher = InboxPublisher.from_config(config)
    else:
        print("[gateway] LiveKit not configured; /session + inbox status push disabled")

    import uvicorn  # noqa: PLC0415

    app = create_app(config, driver=driver, publisher=publisher, token_store=token_store)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
