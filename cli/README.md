# cli

Runnable harnesses, one per subsystem: `uv run python -m cli.<x>` on the host, or
`$RMSAI cli.<x>` in Docker (see the README's Operations runbook). Each has `--help`.

- **Pipeline:**
  - `gen_synthetic`: synthetic signal/event generation.
  - `ingest`: HDF5 → model → `DeviceEvent`; `--file` or `--dir`, `--emit stdout|bus`, `--checkpoint`; prints a scored summary when events carry ground truth.
  - `real_samples`: `list` / `pick --pick LABEL:N,…`; held-out, labelled real-ECG events from an ecg_sigma package into `data/real/` (`PT9#####` pseudonyms).
  - `consume`: bus consumer; persist + criticality-gated dispatch. `--channel voice|text`, `--caller simulated|livekit`, `--transport sip|webrtc`, `--notifier simulated|twilio`, `--number`, `--once`.
  - `outbound`: single-file outbound loop (`--no-answer`, `--fail-delivery`, `--notifier`).
- **Knowledge base:** `kb_vector`, `kb_upload`, `graph`, `kb`, `kb_route`, `kb_dump`, `kb_eval`, `memory`.
- **Chat and voice:** `text_chat`, `voice` (offline demo), `speech_check`, `voice_worker` (`dev`/`start`; `VOICE_WORKER_ROLE=phone` for the phone worker), `livekit_token`, `dispatch` (`--room`, `--all-inbox`).
- **Companion app:** `gateway`, `inbox_publish`, `inbox_probe`.
- **Telephony:**
  - `sip_setup`: provision the telephony LiveKit's trunks + dispatch rule for `TELEPHONY_CARRIER` (signalwire: outbound + inbound trunk, individual rule; twilio: inbound trunk + callee rule, outbound only on the paid path); `--dry-run`, `--swml`, `--twiml`.
  - `call`: ring `OUTBOUND_CALL_NUMBER` on demand; `--caller livekit` needs `LIVEKIT_SIP_TRUNK_ID` (SignalWire, or Twilio paid).

Note: `consume` and `outbound` take the destination from `--number`, else `OUTBOUND_CALL_NUMBER`; a
real call/SMS with neither refuses to start.
