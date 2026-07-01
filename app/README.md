# app

Companion web app — per-hospital worklist (Phase 9). Static HTML/JS (no build step); LiveKit browser
client from CDN.

- `index.html` / `app.js` — PIN login → `POST /session` → join the inbox room
  `rmsai-inbox-<hospital_id>` → render a live worklist from `event`/`status` data messages
  (live-push-only). `applyMessage(state, msg)` is the pure reducer behind the table.

Served by the gateway (`live/gateway.py`, run via `python -m cli.gateway`), same-origin so the scoped
artifact links in inbox messages resolve here.

- **Acknowledge** — per-row button → `POST /ack`; status reflected on every surface.
- **Inline artifacts** — ECG strip / HR-trend sparkline / report render in the detail panel behind
  short-lived scoped tokens.
- **In-app chat** — click a row to scope the conversation to that event (sends `type:"select"` to the
  worker). Type a question or **hold to talk** (push-to-talk mic); answers come from the voice worker
  (`Handler` → de-id → KB/graph). Asking to *see* an artifact pushes a `type:"show"` message that
  renders it inline. Voice needs the worker running with real STT/TTS (`--extra voice`,
  `STT_BACKEND=whisper`/`TTS_BACKEND=piper`); text chat works regardless.

Out of scope: bedside MQTT live waveforms and camera.
