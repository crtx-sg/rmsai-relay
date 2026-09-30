# live

Phase 9 companion-app server-side glue.

- `inbox.py` — publish critical-event / status notifications into the per-hospital LiveKit inbox
  room (`rmsai-inbox-<hospital_id>`) via the LiveKit server API. Pseudonym-only; artifact bytes
  never ride the data channel — only short-lived, single-event scoped links do.
- `artifact_tokens.py` — Redis-backed mint/verify for those scoped artifact tokens.

- `gateway.py` — FastAPI gateway (run via `cli.gateway`): `POST /session` (PIN → inbox join token;
  dispatches the agent into the inbox room; returns `LIVEKIT_PUBLIC_URL` or `LIVEKIT_URL`),
  `POST /ack`, `POST /artifact-link`, `GET /artifact/{token}`; serves `app/`.

Phone calls never touch `live/`: in the telephony split, the inbox and the app stay on the local
LiveKit.

Out of scope (deferred): bedside MQTT→WebRTC waveform streaming and camera relay.
