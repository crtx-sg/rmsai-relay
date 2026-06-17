"""Outbound-alert hand-off (relay -> LiveKit worker), via Redis.

When the consumer places an outbound LiveKit call for an event, the agent worker that joins the
room runs in a *separate process* and otherwise has no idea which patient/event triggered the
call. The relay writes a small `OutboundAlert` (keyed by the room name == session id) here; the
worker reads it on join and, after the PIN gate, voices that event's alert and scopes follow-ups
to that patient.

Redis (not LiveKit room metadata) carries this so the patient pseudonym + alert text stay on our
own store, consistent with redaction-by-construction (G6). Entries expire (TTL) so a missed call
doesn't leave a stale alert behind.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from common.config import DEFAULT, Config

_KEY = "rmsai:outbound_alert:{sid}"
_DEFAULT_TTL = 900  # 15 min — long enough for retries, short enough to self-clean


@dataclass
class OutboundAlert:
    """What the worker needs to turn a placed call into a grounded, event-specific conversation."""

    session_id: str  # == LiveKit room name the call is dialed into
    patient_ref: str
    event_id: str
    spoken_alert: str  # the report spoken after the PIN gate
    bed: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "OutboundAlert":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls(**json.loads(raw))


class OutboundAlertStore:
    """Redis-backed put/get/delete for `OutboundAlert`, keyed by session id (room name)."""

    def __init__(self, redis_client, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self.redis = redis_client
        self.ttl = ttl_seconds

    @classmethod
    def from_config(cls, config: Config = DEFAULT, ttl_seconds: int = _DEFAULT_TTL) -> "OutboundAlertStore":
        import redis  # noqa: PLC0415

        return cls(redis.Redis.from_url(config.redis_url), ttl_seconds)

    def put(self, alert: OutboundAlert) -> None:
        self.redis.set(_KEY.format(sid=alert.session_id), alert.to_json(), ex=self.ttl)

    def get(self, session_id: str) -> OutboundAlert | None:
        raw = self.redis.get(_KEY.format(sid=session_id))
        return OutboundAlert.from_json(raw) if raw else None

    def delete(self, session_id: str) -> None:
        self.redis.delete(_KEY.format(sid=session_id))
