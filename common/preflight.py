"""Turn an unreachable backing service into an actionable message, not a driver traceback.

The backing stores (redis, neo4j, qdrant, livekit) run in Docker **even when the app itself runs
on the host**, so the common failure mode is "I started the CLIs but not the containers". The
driver's own error for that is ~40 lines of connection-pool internals ending in `Error 111
connecting to localhost:6379` — which says what happened but not what to do. This says what to do.
"""

from __future__ import annotations

_START_HINT = (
    "The backing services run in Docker even when the app runs on the host.\n"
    "  Start them:  make stores-up\n"
    "  Check them:  make stores-check\n"
    "  (equivalently: docker compose -f infra/docker-compose.yml up -d redis neo4j qdrant livekit)"
)


def service_unreachable(name: str, url: str, exc: Exception) -> SystemExit:
    """Build a `SystemExit` naming the service, where it was expected, and how to start it.

    Returned rather than raised so the caller can `raise ... from None` and suppress the driver
    traceback, which is noise once the cause is named.
    """
    return SystemExit(
        f"\n{name} is not reachable at {url}\n"
        f"  ({type(exc).__name__}: {exc})\n\n{_START_HINT}\n"
    )
