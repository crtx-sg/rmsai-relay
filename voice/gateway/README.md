# Voice gateway (SIP → LiveKit)

The telephony edge that bridges a phone call (SIP) into a LiveKit room, where the agent
(`voice/livekit_agent.py`) runs STT → handler → TTS. Two supported fronts:

- **LiveKit SIP** (simplest): LiveKit's built-in SIP service terminates the trunk and drops the
  caller into a room. Configure an inbound trunk + dispatch rule (see `sip-inbound.example.yaml`).
- **Jambonz / Asterisk**: a full SIP application server in front of LiveKit, for carrier trunks,
  IVR, and call control. Point its application webhook at the LiveKit room join.

## POC bring-up (manual — needs real telephony)

```bash
docker compose -f infra/docker-compose.yml --profile later up -d livekit
# configure the SIP trunk (sip-inbound.example.yaml) with your provider creds
# run the agent worker (needs livekit-agents installed):
uv run python -c "from voice.livekit_agent import run_agent; run_agent()"
# call the trunk number from a softphone -> you should hear your words echoed back.
```

This end-to-end path (real audio, barge-in over RTP) is **verified manually** — the offline test
suite proves the turn-taking/barge-in/latency logic with stub adapters (`tests/test_voice.py`).

## Outbound (Phase 7)

The same room is used for outbound: the orchestrator creates a SIP participant dialing
`OUTBOUND_CALL_NUMBER`, then speaks the event report and takes follow-ups.
