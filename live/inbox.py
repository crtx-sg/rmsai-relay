"""Per-hospital inbox publisher (Phase 9).

The orchestrator pushes a notification into the facility inbox room `rmsai-inbox-<hospital_id>` for
**every** critical event when `DISPATCH_MODE` includes `app`, and pushes a status message when an
event is acknowledged. The companion app, joined to that room, maintains a live worklist from these.

Design constraints (project rules):
* Published **directly via the LiveKit server API** (RoomService.send_data), NOT by the voice
  worker — so the worklist is live even when no chat/worker session is active.
* **Pseudonym-only.** The message carries `patient` (a `PT####` pseudonym), bed/unit, event type,
  time, criticality, status, and per-kind scoped artifact *links* — never names, notes, or bytes.
* Artifact bytes never ride the data channel; only `{url, token, expires}` links do.

Message schema (JSON on the data channel):
    {type:"event"|"status", event_id, patient, bed, unit, event_type, ts, criticality, status,
     artifact_kinds:[...], links:{<kind>:{url, token, expires}}}

`InboxPublisher` takes an injectable `send_fn(room, payload_bytes)` so tests exercise the schema +
routing without a live LiveKit server; `InboxPublisher.from_config` wires the real SDK send path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from common.config import DEFAULT, Config
from live.artifact_tokens import ArtifactTokenStore

# A published `patient` field must be a pseudonym, never PHI. Enforced by construction (fail-closed):
# publishing anything that isn't a `PT####`-style pseudonym raises rather than leaking a name.
_PSEUDONYM_RE = re.compile(r"^[A-Za-z]{1,4}\d+$")


def inbox_room(config: Config = DEFAULT) -> str:
    """The facility worklist room name for this deployment's hospital."""
    return f"rmsai-inbox-{config.hospital_id}"


def artifact_kinds_for(event, *, config: Config = DEFAULT) -> list[str]:
    """Which already-materialized artifacts this event has, in a stable order.

    * `report` — always (every event gets a materialized markdown report).
    * `ecg_strip` — a strip PNG is (or will be) rendered when there's a producer path or raw signals.
    * `hr_trend` — there's HR history to chart.
    Mirrors what `orchestrator.event_flow.process_device_event` persists on the graph node.
    """
    w = event.window
    kinds = ["report"]
    if w.ecg_plot_ref or (w.signals and config.ecg_plot_enabled):
        kinds.append("ecg_strip")
    if w.vitals_history.get("HR"):
        kinds.append("hr_trend")
    # Stable, schema-documented order.
    order = {"ecg_strip": 0, "hr_trend": 1, "report": 2}
    return sorted(kinds, key=lambda k: order[k])


def mint_artifact_links(
    event_id: str, kinds: list[str], token_store: ArtifactTokenStore, *, now: int | None = None
) -> dict[str, dict[str, Any]]:
    """Mint a scoped token per kind and build the `links` map. URLs are gateway-relative."""
    links: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        token, expires = token_store.mint(event_id, kind, now=now)
        links[kind] = {"url": f"/artifact/{token}", "token": token, "expires": expires}
    return links


def build_event_message(
    *,
    event_id: str,
    patient: str,
    unit: str,
    bed: str,
    event_type: str,
    ts: float,
    criticality: str,
    status: str,
    links: dict[str, dict[str, Any]],
) -> dict:
    """Assemble a `type:"event"` worklist notification. Fails closed if `patient` isn't a pseudonym."""
    if not _PSEUDONYM_RE.match(patient or ""):
        raise ValueError(f"refusing to publish non-pseudonym patient ref: {patient!r}")
    return {
        "type": "event",
        "event_id": event_id,
        "patient": patient,
        "bed": bed,
        "unit": unit,
        "event_type": event_type,
        "ts": ts,
        "criticality": criticality,
        "status": status,
        "artifact_kinds": list(links.keys()),
        "links": links,
    }


def build_status_message(event_id: str, status: str) -> dict:
    """Assemble a `type:"status"` message (e.g. an acknowledge) for the worklist to reflect."""
    return {"type": "status", "event_id": event_id, "status": status}


class InboxPublisher:
    """Publishes worklist `event`/`status` data messages into the per-hospital inbox room."""

    def __init__(self, room: str, *, send_fn: Callable[[str, bytes], None]) -> None:
        self.room = room
        self._send_fn = send_fn

    @classmethod
    def from_config(cls, config: Config = DEFAULT) -> "InboxPublisher":
        """Wire the real LiveKit server-API send path (RoomService.send_data, RELIABLE)."""
        return cls(inbox_room(config), send_fn=_livekit_send_fn(config))

    def _send(self, message: dict) -> None:
        self._send_fn(self.room, json.dumps(message).encode("utf-8"))

    def publish_event(self, message: dict) -> None:
        self._send(message)

    def publish_status(self, event_id: str, status: str) -> None:
        self._send(build_status_message(event_id, status))


def _server_api_url(url: str) -> str:
    """The LiveKit *server API* (RoomService/twirp) is HTTP, not the WebSocket signaling URL.
    Convert `ws(s)://` -> `http(s)://` so `send_data` hits the API, not the RTC socket."""
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    return url


def _livekit_send_fn(config: Config = DEFAULT) -> Callable[[str, bytes], None]:  # pragma: no cover - needs SDK + live server
    """Return a `send_fn(room, data)` that publishes a RELIABLE data message via the LiveKit API."""
    from voice.livekit_cloud import is_configured  # noqa: PLC0415

    if not is_configured(config):
        raise ValueError("LiveKit is not configured (url/key/secret)")

    def _send(room: str, data: bytes) -> None:
        import asyncio  # noqa: PLC0415

        from livekit import api  # noqa: PLC0415
        from livekit.protocol.models import DataPacket  # noqa: PLC0415
        from livekit.protocol.room import SendDataRequest  # noqa: PLC0415

        async def _go():
            lk = api.LiveKitAPI(
                url=_server_api_url(config.livekit_url),
                api_key=config.livekit_api_key,
                api_secret=config.livekit_api_secret,
            )
            try:
                # Do NOT create the room here: pre-creating it empty makes LiveKit fire agent
                # auto-dispatch before any worker/app is present, so the chat worker never joins.
                # The worklist is live-push-only — the app must be connected (so the room exists);
                # if it isn't, send_data raises and the caller treats the push as best-effort.
                await lk.room.send_data(
                    SendDataRequest(room=room, data=data, kind=DataPacket.Kind.RELIABLE)
                )
            finally:
                await lk.aclose()

        asyncio.run(_go())

    return _send
