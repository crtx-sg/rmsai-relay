"""Scoped, short-lived artifact tokens (Phase 9), Redis-backed.

An inbox notification never carries artifact bytes — it carries a *link* per artifact kind, each
guarded by an opaque token that is scoped to exactly one `(event_id, kind)` and expires quickly.
The companion app fetches bytes from the artifact endpoint behind that token; the endpoint calls
`verify` and refuses anything unknown/expired/mismatched (audited).

Mirrors the Redis pattern in `voice/outbound_alert.py` (keyed value + TTL, so a token self-cleans
even if never used). The token is a random opaque id — it embeds no PHI and no event id, so the
value on the wire discloses nothing on its own.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass

from common.config import DEFAULT, Config

_KEY = "rmsai:artifact_token:{tok}"
_DEFAULT_TTL = 300  # 5 min — long enough to open + render, short enough to limit a leaked link

#: Artifact kinds the app can request; the endpoint resolves each from the graph/disk at serve time.
ARTIFACT_KINDS = ("ecg_strip", "hr_trend", "report")


@dataclass(frozen=True)
class ArtifactGrant:
    """What a valid token resolves to: a single event + a single artifact kind."""

    event_id: str
    kind: str


class ArtifactTokenStore:
    """Redis-backed mint/verify for single-`(event_id, kind)` scoped artifact tokens."""

    def __init__(self, redis_client, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self.redis = redis_client
        self.ttl = ttl_seconds

    @classmethod
    def from_config(
        cls, config: Config = DEFAULT, ttl_seconds: int = _DEFAULT_TTL
    ) -> "ArtifactTokenStore":
        import redis  # noqa: PLC0415

        return cls(redis.Redis.from_url(config.redis_url), ttl_seconds)

    def mint(
        self, event_id: str, kind: str, *, ttl: int | None = None, now: int | None = None
    ) -> tuple[str, int]:
        """Mint an opaque token for `(event_id, kind)`. Returns `(token, expires_epoch)`."""
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"unknown artifact kind: {kind!r}")
        ttl = self.ttl if ttl is None else ttl
        issued = int(time.time() if now is None else now)
        expires = issued + ttl
        token = secrets.token_urlsafe(24)
        record = json.dumps({"event_id": event_id, "kind": kind, "exp": expires})
        self.redis.set(_KEY.format(tok=token), record, ex=ttl)
        return token, expires

    def verify(self, token: str, kind: str | None = None) -> ArtifactGrant | None:
        """Resolve a token to its `(event_id, kind)` grant, or `None` if unknown/expired/mismatched.

        Redis TTL handles expiry (the key is simply gone). When `kind` is given it must match the
        minted kind — a token scoped to one artifact kind can't be replayed for another.
        """
        if not token:
            return None
        raw = self.redis.get(_KEY.format(tok=token))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        rec = json.loads(raw)
        if kind is not None and rec.get("kind") != kind:
            return None
        return ArtifactGrant(event_id=rec["event_id"], kind=rec["kind"])
