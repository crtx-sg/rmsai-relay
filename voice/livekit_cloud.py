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


def _http_url(url: str) -> str:
    """The LiveKit *server API* (RoomService/AgentDispatch/twirp) is HTTP, not the WS signal URL."""
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    return url


#: A LiveKit participant with `kind == AGENT` (protocol ParticipantInfo.Kind.AGENT).
_AGENT_KIND = 4


#: LiveKit `JobStatus` enum values that mean a job is still live (livekit.api.JobStatus).
_JS_PENDING, _JS_RUNNING = 0, 1
#: A dispatched agent joins within ~40s (cold-start: STT/TTS/VAD/orchestrator load). A PENDING/RUNNING
#: job newer than this is "an agent is coming"; an OLDER one whose agent never became a participant is
#: a zombie (dead/removed worker — LiveKit leaves the job RUNNING) and must NOT block re-dispatch.
_JOB_FRESH_NS = 75 * 1_000_000_000


def _has_fresh_active_job(dispatches, *, now_ns: int, fresh_ns: int = _JOB_FRESH_NS) -> bool:
    """Pure: True if any dispatch has a PENDING/RUNNING job that started within `fresh_ns` of now.

    "Fresh + active" = an agent just dispatched and still cold-starting, so a concurrent dispatch
    (gateway `/session` + worker auto-redispatch) would add a *second* agent and every reply would be
    spoken twice (an echo). Recency is essential: a RUNNING job left behind by a dead/removed worker
    keeps its status forever but has an OLD `started_at`, so it ages out and stops blocking — which is
    what lets a worker restart re-wire the room. `started_at` is unix nanoseconds; a job with no
    timestamp (0) is treated as just-created (fresh). Unit-tested with duck-typed fakes; real inputs
    are LiveKit proto `AgentDispatch` objects.
    """
    for d in dispatches:
        for j in getattr(getattr(d, "state", None), "jobs", None) or []:
            js = getattr(j, "state", None)
            if getattr(js, "status", None) not in (_JS_PENDING, _JS_RUNNING):
                continue
            started = getattr(js, "started_at", 0) or 0
            if started == 0 or (now_ns - started) < fresh_ns:
                return True
    return False


async def _room_has_agent_or_pending(lk, api, room: str) -> bool:  # pragma: no cover - live SDK
    """True if `room` has an agent participant OR a **fresh** pending/running agent-dispatch job.

    The fresh-job check catches an agent dispatched but not yet joined (cold-start) so we never
    double-dispatch (echo), while ignoring zombie RUNNING jobs from dead workers so a restart can
    re-wire. Best-effort: any lookup error is treated as "no agent" (fail toward dispatching, since a
    missing agent is the failure we're fixing)."""
    import time  # noqa: PLC0415

    try:
        parts = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        if any(p.kind == _AGENT_KIND for p in parts.participants):
            return True
    except Exception:  # noqa: BLE001 - room not found yet -> no agent present
        pass
    try:
        disp = await lk.agent_dispatch.list_dispatch(room_name=room)
        return _has_fresh_active_job(disp, now_ns=time.time_ns())
    except Exception:  # noqa: BLE001 - no dispatch history / transient -> treat as none
        return False


def create_agent_dispatch(
    room: str, *, config: Config = DEFAULT, agent_name: str | None = None, metadata: str = "",
) -> bool:  # pragma: no cover - needs the SDK + a live LiveKit server
    """Explicitly dispatch the named agent worker into `room`. Returns True if a dispatch was created.

    Idempotent by design: if an AGENT participant is already in the room this is a no-op (so repeated
    `/session` calls or reloads don't spawn duplicate agents). If the room doesn't exist yet, the
    dispatch creates it and the agent joins when a worker is available — so this is safe to call
    *before* the app/clinician joins (inbox) or before the SIP call is placed (outbound).

    Fail-closed for duplicates, best-effort for delivery: callers should wrap in try/except — a failed
    dispatch must not break the surrounding action (auth, call placement), just leaves Q&A unwired.
    """
    import asyncio  # noqa: PLC0415

    from livekit import api  # noqa: PLC0415

    name = agent_name or config.livekit_agent_name

    async def _go() -> bool:
        lk = api.LiveKitAPI(
            url=_http_url(config.livekit_url),
            api_key=config.livekit_api_key,
            api_secret=config.livekit_api_secret,
        )
        try:
            # Skip if an agent is already present OR a dispatch job is pending/running (an agent
            # that's been dispatched but is still cold-starting). Prevents a duplicate agent (echo).
            if await _room_has_agent_or_pending(lk, api, room):
                return False
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(room=room, agent_name=name, metadata=metadata)
            )
            return True
        finally:
            await lk.aclose()

    return asyncio.run(_go())


def _should_dispatch(
    room_name: str,
    participant_kinds,
    *,
    prefix: str,
    require_human: bool,
    skip=frozenset(),
) -> bool:
    """Pure predicate: should the agent be dispatched into `room_name`? (see redispatch_existing_rooms)

    True iff the room matches `prefix`, is not in `skip` (a per-run cooldown set), has **no** agent
    already, and — when `require_human` — has at least one non-agent participant (don't spawn an
    agent into an empty/stale room). `participant_kinds` is the room's `ParticipantInfo.Kind` list.
    """
    if room_name in skip:
        return False
    if prefix and not room_name.startswith(prefix):
        return False
    if _AGENT_KIND in participant_kinds:
        return False  # an agent is already serving this room
    if require_human and not any(k != _AGENT_KIND for k in participant_kinds):
        return False  # nobody to serve
    return True


def redispatch_existing_rooms(
    config: Config = DEFAULT,
    *,
    prefix: str = "rmsai-inbox-",
    require_human: bool = True,
    skip=frozenset(),
    agent_name: str | None = None,
) -> list[str]:  # pragma: no cover - needs the SDK + a live LiveKit server
    """Dispatch the agent into every live room matching `prefix` that currently lacks an agent.

    Backs worker-startup auto-redispatch (a restarted worker re-joins live inbox rooms without an app
    re-login) and `cli.dispatch --all-inbox`. Idempotent: rooms that already have an agent (or are in
    `skip`) are left alone. Returns the list of room names a dispatch was created for.

    Best-effort: any failure (no server, SDK missing) yields `[]` rather than raising, so a worker
    start never fails on this.
    """
    import asyncio  # noqa: PLC0415

    try:
        from livekit import api  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - livekit extra not installed -> nothing to do
        return []

    name = agent_name or config.livekit_agent_name

    async def _go() -> list[str]:
        lk = api.LiveKitAPI(
            url=_http_url(config.livekit_url),
            api_key=config.livekit_api_key,
            api_secret=config.livekit_api_secret,
        )
        out: list[str] = []
        try:
            rooms = await lk.room.list_rooms(api.ListRoomsRequest())
            for r in rooms.rooms:
                try:
                    parts = await lk.room.list_participants(
                        api.ListParticipantsRequest(room=r.name)
                    )
                    kinds = [p.kind for p in parts.participants]
                except Exception:  # noqa: BLE001 - transient; treat as unknown, skip this room
                    continue
                if not _should_dispatch(r.name, kinds, prefix=prefix, require_human=require_human,
                                        skip=skip):
                    continue
                # Authoritative dedupe: skip if a dispatch job is already pending/running (e.g. the
                # gateway /session dispatch is in flight and the agent is still cold-starting) — else
                # we'd add a second agent and every reply would be spoken twice.
                if await _room_has_agent_or_pending(lk, api, r.name):
                    continue
                await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(room=r.name, agent_name=name)
                )
                out.append(r.name)
        finally:
            await lk.aclose()
        return out

    try:
        return asyncio.run(_go())
    except Exception:  # noqa: BLE001 - never let a background/CLI redispatch crash the caller
        return []


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def verify_access_token(token: str, config: Config = DEFAULT, *, now: int | None = None) -> dict | None:
    """Verify a LiveKit HS256 token minted by `access_token` (our API secret). Returns the payload
    dict when the signature is valid and it is within nbf/exp, else `None`.

    Used to authorize companion-app HTTP calls (e.g. `/ack`): possessing a valid inbox token proves
    the caller cleared the PIN gate, since tokens are only minted after `POST /session` succeeds.
    """
    if not token or not config.livekit_api_secret:
        return None
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    signing_input = f"{header_b64}.{payload_b64}"
    expected = _b64url(
        hmac.new(config.livekit_api_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(expected, sig_b64):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    ts = int(now if now is not None else time.time())
    if int(payload.get("exp", 0)) < ts or int(payload.get("nbf", 0)) > ts:
        return None
    return payload


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


def build_sip_participant_kwargs(
    *, room: str, number: str, config: Config = DEFAULT, trunk_id: str | None = None,
    identity: str = "clinician", name: str = "rmsai-alert", wait_until_answered: bool = True,
) -> dict:
    """Fields for `CreateSIPParticipantRequest`, as a plain dict so they can be asserted offline.

    Split out from the SDK call because the request is where a real trunk rejects us and where a
    runaway call gets expensive, and neither is something to discover on a live carrier:

    * `sip_number` is the caller ID (`OUTBOUND_FROM`). Most carrier trunks drop a call presenting no
      valid from-number, and it is what shows on the clinician's handset — omitting it is why an
      otherwise correct trunk config still fails to ring.
    * `ringing_timeout` bounds how long an unanswered call holds a trunk channel.
    * `max_call_duration` is the backstop for a call answered by voicemail, which otherwise bills
      until somebody notices.

    Raises `ValueError` when no trunk is configured — fail fast rather than let the SDK report it as
    a generic dial failure that the retry policy would then dutifully repeat.
    """
    trunk = trunk_id or config.livekit_sip_trunk_id
    if not trunk:
        raise ValueError("LIVEKIT_SIP_TRUNK_ID is not configured")
    kwargs = {
        "sip_trunk_id": trunk,
        "sip_call_to": number,
        "room_name": room,
        "participant_identity": identity,
        "participant_name": name,
        "wait_until_answered": wait_until_answered,
        "ringing_timeout": _duration(config.sip_ringing_timeout_s),
        "max_call_duration": _duration(config.sip_max_call_duration_s),
    }
    if config.outbound_from:  # caller ID; omitted entirely when unset so the trunk default applies
        kwargs["sip_number"] = config.outbound_from
    return kwargs


def _duration(seconds: int):
    """Seconds -> protobuf `Duration` (the SIP request's timeout fields), or None to leave unset."""
    if not seconds or seconds <= 0:
        return None
    from google.protobuf.duration_pb2 import Duration  # noqa: PLC0415

    return Duration(seconds=int(seconds))


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

        kwargs = build_sip_participant_kwargs(
            room=room, number=number, config=self.config, trunk_id=trunk_id,
            identity=identity, name=name, wait_until_answered=wait_until_answered,
        )

        async def _go():
            lk = self._api()
            try:
                return await lk.sip.create_sip_participant(CreateSIPParticipantRequest(**kwargs))
            finally:
                await lk.aclose()

        return self._run(_go())
