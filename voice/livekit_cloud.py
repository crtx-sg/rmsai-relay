"""LiveKit Cloud integration — access tokens + room/SIP operations.

Works against either a self-hosted server (`ws://localhost:7880`) or **LiveKit Cloud**
(`wss://<project>.livekit.cloud`); the URL, API key, and API secret are all configurable
(`LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`).

* `access_token(...)` builds a LiveKit join token — an HS256 JWT (iss=API key, sub=identity, a
  `video` grant) signed with the API secret. Pure-stdlib, so it is unit-tested offline and the
  emitted token is exactly what LiveKit Cloud accepts.
* `LiveKitClient` wraps the official `livekit-api` SDK (lazy import) for room creation and
  outbound SIP dialing (`CreateSIPParticipant`). Needs network + the SDK; verified against Cloud.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from common.config import DEFAULT, Config


def is_configured(config: Config = DEFAULT) -> bool:
    """True when a LiveKit endpoint + API key + secret are all set."""
    return bool(config.livekit_url and config.livekit_api_key and config.livekit_api_secret)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def access_token(
    *,
    identity: str,
    room: str,
    config: Config = DEFAULT,
    name: str | None = None,
    ttl_seconds: int = 3600,
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True,
    room_admin: bool = False,
    metadata: str | None = None,
    now: int | None = None,
) -> str:
    """Return a signed LiveKit join token (HS256 JWT) for `identity` to join `room`."""
    if not config.livekit_api_key or not config.livekit_api_secret:
        raise ValueError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not configured")
    issued = int(now if now is not None else time.time())

    grant = {
        "room": room,
        "roomJoin": True,
        "canPublish": can_publish,
        "canSubscribe": can_subscribe,
        "canPublishData": can_publish_data,
    }
    if room_admin:
        grant["roomAdmin"] = True

    payload: dict = {
        "iss": config.livekit_api_key,
        "sub": identity,
        "nbf": issued,
        "exp": issued + ttl_seconds,
        "video": grant,
    }
    if name:
        payload["name"] = name
    if metadata:
        payload["metadata"] = metadata

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    sig = hmac.new(config.livekit_api_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


class LiveKitClient:  # pragma: no cover - needs the SDK + a live LiveKit endpoint
    """Room + SIP operations via the official `livekit-api` SDK (lazy)."""

    def __init__(self, config: Config = DEFAULT) -> None:
        if not is_configured(config):
            raise ValueError("LiveKit is not configured (url/key/secret)")
        self.config = config

    def _run(self, coro):
        import asyncio  # noqa: PLC0415

        return asyncio.run(coro)

    def _api(self):
        from livekit import api  # noqa: PLC0415

        return api.LiveKitAPI(
            url=self.config.livekit_url,
            api_key=self.config.livekit_api_key,
            api_secret=self.config.livekit_api_secret,
        )

    def create_outbound_sip_call(
        self, *, room: str, number: str, identity: str = "clinician",
        name: str = "rmsai-alert", trunk_id: str | None = None, wait_until_answered: bool = True,
    ):
        """Dial `number` into `room` via the outbound SIP trunk. Returns SIPParticipantInfo."""
        from livekit.protocol.sip import CreateSIPParticipantRequest  # noqa: PLC0415

        trunk = trunk_id or self.config.livekit_sip_trunk_id
        if not trunk:
            raise ValueError("LIVEKIT_SIP_TRUNK_ID is not configured")

        async def _go():
            lk = self._api()
            try:
                return await lk.sip.create_sip_participant(
                    CreateSIPParticipantRequest(
                        sip_trunk_id=trunk, sip_call_to=number, room_name=room,
                        participant_identity=identity, participant_name=name,
                        wait_until_answered=wait_until_answered,
                    )
                )
            finally:
                await lk.aclose()

        return self._run(_go())
