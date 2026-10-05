# cli

Runnable harnesses, one per subsystem: `uv run python -m cli.<x>` on the host, or
`$RMSAI cli.<x>` in Docker (see the README's Operations runbook). Each has `--help`.

- **Pipeline:**
  - `gen_synthetic`: synthetic signal/event generation.
  - `ingest`: HDF5 → model → `DeviceEvent`; `--file` or `--dir`, `--emit stdout|bus`, `--checkpoint`; each line has an `outcome`; `--explain` adds `why`, `--metrics` the full breakdown.
  - `real_samples`: `list` / `pick --pick LABEL:N,…`; held-out, labelled real-ECG events from an ecg_sigma package into `data/real/` (`PT9#####` pseudonyms).
  - `consume`: bus consumer; persist + criticality-gated dispatch. `--channel voice|text`, `--caller simulated|livekit`, `--transport sip|webrtc`, `--notifier simulated|twilio`, `--number`, `--once`, `--perf-every N`, `--trend-samples`. Logs an `[event]` line per event and a `[perf]` summary.
  - `model_perf`: model-performance summary on request from the graph (`--since`, `--model`, `--dataset`, `--events`, `--json`).
  - `outbound`: single-file outbound loop (`--no-answer`, `--fail-delivery`, `--notifier`).
- **Knowledge base:** `kb_vector`, `kb_upload`, `graph`, `kb`, `kb_route`, `kb_dump`, `kb_eval`, `memory`.
- **Chat and voice:** `text_chat`, `voice` (offline demo), `speech_check`, `voice_worker` (`dev`/`start`; `VOICE_WORKER_ROLE=phone` for the phone worker), `livekit_token`, `dispatch` (`--room`, `--all-inbox`).
- **Companion app:** `gateway`, `inbox_publish`, `inbox_probe`.
- **Telephony:**
  - `sip_setup`: provision the telephony LiveKit's trunks + dispatch rule for `TELEPHONY_CARRIER` (signalwire: outbound + inbound trunk, individual rule; twilio: inbound trunk + callee rule, outbound only on the paid path); `--dry-run`, `--swml`, `--twiml`.
  - `call`: ring the hospital's `outbound.call_number` on demand; `--caller livekit` needs `LIVEKIT_SIP_TRUNK_ID` (SignalWire, or Twilio paid).

Note: `consume` and `outbound` take the destination from `--number`, else the hospital's
`outbound.call_number` (`config/hospitals/<HOSPITAL_ID>.yaml`; `OUTBOUND_CALL_NUMBER` overrides it); a
real call/SMS with neither refuses to start. `text_chat --llm` accepts `echo`, `ollama`, `anthropic`,
`gemini`, `openai`.
