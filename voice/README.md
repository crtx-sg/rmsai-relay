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
- `sip_setup.py`: plan/apply the telephony SIP objects (inbound trunk, callee rule, optional outbound trunk) and build the TwiML for the Twilio bridge.
- `gateway/`: telephony-edge notes.

**Telephony split:** phone calls run on a second LiveKit (LiveKit Cloud, `LIVEKIT_SIP_URL`), with
Twilio Programmable Voice forwarding the phone leg in via TwiML `<Dial><Sip>`. Run the phone worker
with `make phone-up`. Call-in is set up; event-driven outbound via the Twilio Calls API is the next
phase. See the README's Phone calls section.
