# cli

Runnable harnesses, one per subsystem: `uv run python -m cli.<x>` on the host, or
`$RMSAI cli.<x>` in Docker (see the README's Operations runbook). Each has `--help`.

- **Pipeline:**
  - `gen_synthetic`: synthetic signal/event generation.
  - `ingest`: HDF5 → model → `DeviceEvent`; `--file` or `--dir`, `--emit stdout|bus`, `--checkpoint`; each line has an `outcome`; `--explain` adds `why`, `--metrics` the full breakdown.
  - `real_samples`: `list` / `pick --pick LABEL:N,…`; held-out, labelled real-ECG events from an ecg_sigma package into `data/real/` (`PT9#####` pseudonyms).
  - `consume`: bus consumer; persist + criticality-gated dispatch. `--channel voice|text`, `--caller simulated|livekit`, `--transport sip|webrtc`, `--notifier simulated|signalwire`, `--number`, `--once`, `--perf-every N`, `--trend-samples`. Logs an `[event]` line per event and a `[perf]` summary. With `--transport sip`, calls go through the Redis call queue (`orchestrator/call_queue.py`): most severe first, one at a time (`[queued]` / `[calls]` lines).
  - `model_perf`: model-performance summary on request from the graph (`--since`, `--model`, `--dataset`, `--events`, `--json`).
  - `events`: event history from the graph (`--patient`, `--bed`, `--unit`, `--type`, `--status any|pending|acknowledged|resolved|withheld`, `--min-criticality`, `--since`, `--labelled`, `--json`); `--ask "<question>"` (+ `--as-patient`, `--spoken`) shows the query a voice/chat question runs and the answer. Read-only.
  - `outbound`: single-file outbound loop (`--no-answer`, `--fail-delivery`, `--notifier`).
- **Knowledge base:** `kb_vector`, `kb_upload`, `graph`, `kb`, `kb_route`, `kb_dump`, `kb_eval`, `memory`.
- **Chat and voice:** `text_chat`, `voice` (offline demo), `speech_check`, `voice_worker` (`dev`/`start`; `VOICE_WORKER_ROLE=phone` for the phone worker), `livekit_token`, `dispatch` (`--room`, `--all-inbox`).
- **Companion app:** `gateway`, `inbox_publish`, `inbox_probe`.
- **Telephony:**
  - `sip_setup`: provision the telephony LiveKit's SignalWire objects (outbound + inbound trunk, individual call-in rule); `--dry-run`, `--swml`.
  - `call`: ring the hospital's `outbound.call_number` on demand; `--caller livekit` needs `LIVEKIT_SIP_TRUNK_ID` (the SignalWire outbound trunk).

Note: `consume` and `outbound` take the destination from `--number`, else the hospital's
`outbound.call_number` (`config/hospitals/<HOSPITAL_ID>.yaml`; `OUTBOUND_CALL_NUMBER` overrides it); a
real call/SMS with neither refuses to start. `text_chat --llm` accepts `echo`, `ollama`, `anthropic`,
`gemini`, `openai`.
