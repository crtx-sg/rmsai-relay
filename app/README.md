# app

Companion web app — per-hospital worklist (Phase 9). Static HTML/JS (no build step); LiveKit browser
client vendored in `vendor/` (`livekit-client.umd.min.js`).

- `index.html` / `app.js` — PIN login → `POST /session` → join the inbox room
  `rmsai-inbox-<hospital_id>` → render a live worklist from `event`/`status` data messages
  (live-push-only). `applyMessage(state, msg)` is the pure reducer behind the table.

Served by the gateway (`live/gateway.py`, run via `python -m cli.gateway`), same-origin so the scoped
artifact links in inbox messages resolve here.

- **Acknowledge** — per-row button → `POST /ack`; status reflected on every surface.
- **Inline artifacts** — ECG strip / HR-trend sparkline / report render in the detail panel behind
  short-lived scoped tokens; a click mints a fresh link via `POST /artifact-link`.
- **In-app chat** — click a row to scope the conversation to that event (sends the chat text
  `/select <event_id>` on topic `lk.chat` to the worker). Type a question or **hold to talk** (push-to-talk mic); answers come from the voice worker
  (`Handler` → de-id → KB/graph). Asking to *see* an artifact pushes a `type:"show"` message that
  renders it inline. Voice needs the worker running with real STT/TTS (`--extra voice`,
  `STT_BACKEND=whisper`/`TTS_BACKEND=piper`); text chat works regardless.
- **Speak-on-select** — with `INBOX_SPEAK_ON_SELECT=true` (default), selecting a row also **speaks**
  that event's stored report summary aloud via TTS (in addition to scoping chat). No wake word / mic
  needed; disable to keep selection silent.

- **Why + source + outcome** — each row shows why it's on the worklist (`why`), a data-source badge
  for demo data (`source`), and, for labelled data, the outcome against the truth. Selecting a row
  loads the full explanation and provenance from `POST /event-info`.
- **Model performance tab** — `POST /metrics` (session-gated): per-model tiles with 95 % intervals,
  confusion matrix, per-class table, and every labelled event (alerted or not), refreshing as events
  arrive. Rendering is tested in Node against a stub DOM (`tests/test_app_render.py`).

**Stale-cache check:** the build tag (`APP_BUILD` in `app.js`) shows on screen; if it's old, the
browser is running a cached `app.js`.

Before publishing events, log in: the worklist is live-push-only (no backlog), so events pushed
while the app is closed never appear.

Out of scope: bedside MQTT live waveforms and camera.
