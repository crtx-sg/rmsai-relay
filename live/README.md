# live

Phase 9 companion-app server-side glue.

- `inbox.py` — publish critical-event / status notifications into the per-hospital LiveKit inbox
  room (`rmsai-inbox-<hospital_id>`) via the LiveKit server API. Pseudonym-only; artifact bytes
  never ride the data channel — only short-lived, single-event scoped links do.
- `artifact_tokens.py` — Redis-backed mint/verify for those scoped artifact tokens.

Later Phase 9 steps add the FastAPI gateway (session/PIN, `/artifact`, `/ack`) and the static app.

Out of scope (deferred): bedside MQTT→WebRTC waveform streaming and camera relay.
