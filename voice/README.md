# voice

Voice surfaces: LiveKit ↔ STT/TTS, the PIN auth gate, inbound and outbound calling, SIP provisioning.
Phases 5–7, plus the telephony split.

- `adapters.py`: STT/TTS behind `STTAdapter`/`TTSAdapter`: stub, whisper, piper, elevenlabs (cloud: synthetic data only); `speakable`.
- `auth.py` (shared-PIN gate) · `wake.py` ("hey vios" wake word) · `session.py` (turn-taking, barge-in).
- `handlers.py`: Inbox, Outbound and PIN-gated Q&A handlers.
- `livekit_agent.py`: the agent worker; `resolve_handler` picks the handler per room. `worker_config` picks the role: `app` (default) or `phone` (telephony LiveKit, `rmsai-agent-phone`).
- `livekit_cloud.py`: tokens, `create_agent_dispatch`, `redispatch_existing_rooms`, SIP participant (dial).
- `outbound.py`: `Caller`, `LiveKitCaller`, retry policy, `place_predefined_call`, number masking.
- `outbound_alert.py`: the Redis hand-off of a staged alert from the consumer to the worker.
- `sip_setup.py`: plan/apply the telephony SIP objects for SignalWire: outbound + inbound trunk, individual call-in rule, and the SWML scripts.
- `gateway/`: telephony-edge notes.

**Telephony split:** phone calls run on a second LiveKit (LiveKit Cloud, `LIVEKIT_SIP_URL`). The
carrier is a native SignalWire SIP trunk both ways. Run the phone worker with `make phone-up`. Set-up: `TELEPHONY_SETUP.md`; call flows:
`ARCHITECTURE.md` § Telephony split; engineering notes: `gateway/README.md`.
