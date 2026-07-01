"""Phase 9 companion-app gateway (FastAPI).

Serves the static worklist app and the small authenticated endpoints it needs, all same-origin so
the artifact links in inbox messages (`/artifact/<token>`) resolve against this gateway.

Step 2 adds:
* `POST /session {pin}` — the PIN gate. On success it mints a **per-hospital inbox join token**
  (a LiveKit JWT scoped to `rmsai-inbox-<hospital_id>`, reusing `voice.livekit_cloud.access_token`)
  so the app can join the facility worklist room. Every attempt is audited.
* static mount of `app/` (the worklist SPA).

Later steps mount `POST /ack` and `GET /artifact/<token>` onto the same app.

`create_app(...)` is a factory so tests drive it with `fastapi.testclient.TestClient` and an
injected `AuditLog`; FastAPI is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from pydantic import BaseModel

from common.audit import AuditLog
from common.config import DEFAULT, Config
from kb.graph.events import get_event_artifacts, get_event_patient, set_event_status
from live.inbox import inbox_room
from voice.auth import PinAuthGate
from voice.livekit_cloud import access_token, is_configured, verify_access_token


def _resolve_within(base: str, path: str):
    """Resolve `path` and confirm it stays under `base` (defense against a bad stored ref)."""
    from pathlib import Path as _P  # noqa: PLC0415

    base_r = _P(base).resolve()
    target = _P(path).resolve()
    if base_r != target and base_r not in target.parents:
        return None
    return target if target.is_file() else None

_APP_DIR = Path(__file__).resolve().parents[1] / "app"


class SessionRequest(BaseModel):
    """`POST /session` body. Defined at module scope so FastAPI resolves the body annotation
    (postponed annotations turn a factory-local class into an unresolvable string)."""

    pin: str


class AckRequest(BaseModel):
    """`POST /ack` body. `session` is the inbox join token from `POST /session` (proof of PIN)."""

    event_id: str
    session: str


def create_app(
    config: Config = DEFAULT,
    *,
    audit: AuditLog | None = None,
    driver=None,
    publisher=None,
    token_store=None,
    app_dir: Path | None = None,
):
    """Build the gateway FastAPI app.

    `driver` (a `GraphDriver`) + `publisher` (an `InboxPublisher`) back the acknowledge round-trip;
    `token_store` (an `ArtifactTokenStore`) backs the artifact endpoint. All are injectable so tests
    drive the app with fakes.
    """
    from fastapi import FastAPI, HTTPException  # noqa: PLC0415
    from fastapi.responses import FileResponse, JSONResponse  # noqa: PLC0415
    from fastapi.staticfiles import StaticFiles  # noqa: PLC0415

    audit = audit or AuditLog(config.audit_log_path)
    gate = PinAuthGate(config)
    app_dir = _APP_DIR if app_dir is None else app_dir

    api = FastAPI(title="rmsai companion-app gateway")

    @api.post("/session")
    def session(req: SessionRequest) -> dict:
        """PIN -> a scoped inbox join token. Fail-closed: no token unless the PIN verifies."""
        room = inbox_room(config)
        # subject is the facility (not PHI); never log the PIN.
        subject = config.hospital_id or "-"
        if not gate.verify(req.pin):
            audit.write(actor="app", action="inbox_session", subject=subject, outcome="denied")
            raise HTTPException(status_code=401, detail="invalid PIN")
        if not is_configured(config):
            audit.write(actor="app", action="inbox_session", subject=subject,
                        outcome="unconfigured")
            raise HTTPException(status_code=503, detail="LiveKit is not configured")
        identity = f"clinician-{secrets.token_hex(3)}"  # unique per session (LiveKit needs it)
        token = access_token(
            identity=identity, room=room, name="clinician", config=config,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        )
        audit.write(actor="app", action="inbox_session", subject=subject, outcome="authorized",
                    identity=identity)
        return {
            "url": config.livekit_url,
            "room": room,
            "identity": identity,
            "token": token,
            "hospital_id": config.hospital_id,
        }

    @api.post("/ack")
    def ack(req: AckRequest) -> dict:
        """Acknowledge an event: flip status -> acknowledged, audit it, and push a status message
        back to the inbox so the worklist (and any parallel SIP surface) reflects it everywhere.

        Fail-closed: a bad/expired session token is refused (401); an unknown event is refused (404).
        """
        # 1. AuthN: a valid inbox token (signed by us, unexpired, scoped to this hospital's room).
        payload = verify_access_token(req.session, config)
        room = inbox_room(config)
        if not payload or (payload.get("video") or {}).get("room") != room:
            audit.write(actor="app", action="acknowledgment", subject="-", outcome="unauthorized",
                        event_id=req.event_id, surface="app")
            raise HTTPException(status_code=401, detail="invalid session")
        if driver is None:
            raise HTTPException(status_code=503, detail="event store not configured")

        # 2. Existence check + pseudonym to audit against (never the event uuid as a subject).
        patient = get_event_patient(driver, req.event_id)
        if patient is None:
            audit.write(actor="app", action="acknowledgment", subject="-", outcome="unknown_event",
                        event_id=req.event_id, surface="app")
            raise HTTPException(status_code=404, detail="unknown event")

        # 3. Flip status, audit, and push the change back to every surface (best-effort publish).
        set_event_status(driver, req.event_id, "acknowledged")
        audit.write(actor="app", action="acknowledgment", subject=patient, outcome="acknowledged",
                    event_id=req.event_id, surface="app")
        if publisher is not None:
            try:
                publisher.publish_status(req.event_id, "acknowledged")
            except Exception as exc:  # noqa: BLE001 - status is already persisted; push is best-effort
                print(f"[gateway] ack status push failed for {req.event_id}: {exc}")
        return {"event_id": req.event_id, "status": "acknowledged"}

    @api.get("/artifact/{token}")
    def artifact(token: str):
        """Serve one artifact for a valid, single-event scoped token. Every hit is audited; the
        bytes are only released behind a good token (unknown/expired/missing -> 404, no bytes)."""
        if token_store is None or driver is None:
            raise HTTPException(status_code=503, detail="artifacts not configured")

        grant = token_store.verify(token)
        if grant is None:
            audit.write(actor="app", action="view_artifact", subject="-", outcome="denied",
                        reason="bad_token")
            raise HTTPException(status_code=404, detail="invalid or expired token")

        info = get_event_artifacts(driver, grant.event_id)
        if info is None:
            audit.write(actor="app", action="view_artifact", subject="-", outcome="unknown_event",
                        event_id=grant.event_id, kind=grant.kind)
            raise HTTPException(status_code=404, detail="unknown event")

        patient = info.get("patient") or "-"

        def _served():
            audit.write(actor="app", action="view_artifact", subject=patient, outcome="served",
                        event_id=grant.event_id, kind=grant.kind)

        if grant.kind == "hr_trend":
            _served()
            return JSONResponse({
                "event_id": grant.event_id,
                "hr_history": info.get("hr_history") or [],
                "hr_history_ts": info.get("hr_history_ts") or [],
            })

        if grant.kind == "ecg_strip":
            target = _resolve_within(config.plot_dir, info.get("ecg_plot_ref") or "")
            if target is None:
                audit.write(actor="app", action="view_artifact", subject=patient,
                            outcome="missing", event_id=grant.event_id, kind=grant.kind)
                raise HTTPException(status_code=404, detail="artifact not available")
            _served()
            return FileResponse(str(target), media_type="image/png")

        if grant.kind == "report":
            target = _resolve_within(config.report_dir, info.get("report_uri") or "")
            if target is None:
                audit.write(actor="app", action="view_artifact", subject=patient,
                            outcome="missing", event_id=grant.event_id, kind=grant.kind)
                raise HTTPException(status_code=404, detail="artifact not available")
            _served()
            return FileResponse(str(target), media_type="text/markdown")

        raise HTTPException(status_code=404, detail="unknown artifact kind")

    # Static worklist app last, at "/", so the explicit API routes above take precedence.
    if app_dir.is_dir():
        api.mount("/", StaticFiles(directory=str(app_dir), html=True), name="app")

    return api
