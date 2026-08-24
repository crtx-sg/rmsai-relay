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

## On-demand call (`cli.call`)

`uv run python -m cli.call --caller livekit` rings `OUTBOUND_CALL_NUMBER` without waiting for an
event. Each call gets its own room (`LIVEKIT_CALL_ROOM_PREFIX` + a random id); the agent is
dispatched into it **before** the dial, and because no alert is staged there the worker runs the
PIN-gated Q&A handler — the callee hears the PIN prompt, authenticates, then asks questions.
Needs `LIVEKIT_SIP_TRUNK_ID` (outbound trunk) and `OUTBOUND_FROM` (caller ID). `--caller simulated`
(the default) exercises the whole path with no telephony.

## ⚠ Inbound is not wired up

`sip-inbound.example.yaml` below predates the switch to **explicit** agent dispatch
(`WorkerOptions(agent_name=...)`, see `voice/livekit_agent.py`). A worker registered under a name
does **not** auto-join new rooms, so an inbound call matching `dispatchRuleIndividual` today creates
a room with **no agent in it** — the caller hears silence. Wiring this up (naming the agent in the
dispatch rule's room config, so LiveKit requests it per call) is deliberately out of scope for the
current change; the outbound leg above dispatches explicitly and is unaffected.
