# Voice gateway (phone ↔ LiveKit)

The telephony edge: how a phone call reaches a LiveKit room, where the agent
(`voice/livekit_agent.py`) runs STT → handler → TTS. The full set-up guide is
[README § Phone calls](../../README.md#phone-calls-twilio--livekit-cloud); this file is the
engineering summary.

## Architecture (telephony split)

A LiveKit SIP bridge only places calls into rooms on its **own** server, and the compose LiveKit
(`infra/livekit.yaml`) has no SIP service. So phone calls run on a second LiveKit, a **LiveKit Cloud**
project (`LIVEKIT_SIP_URL`). The app inbox and WebRTC stay local. `Config.telephony()` is the config
the call paths use; the phone worker registers there as `rmsai-agent-phone`
(`VOICE_WORKER_ROLE=phone`, `make phone-up`).

The carrier is **Twilio Programmable Voice**, because a Twilio trial blocks Elastic SIP Trunking.
Twilio handles the phone leg and forwards the answered call into LiveKit Cloud with TwiML
`<Dial><Sip>sip:<room>@<LIVEKIT_SIP_URI></Sip>`, using digest credentials. The SIP user part names
the room.

## Provisioning (`cli.sip_setup`)

```bash
# .env: LIVEKIT_SIP_URL/_API_KEY/_API_SECRET, LIVEKIT_SIP_URI, LIVEKIT_SIP_INBOUND_USERNAME/_PASSWORD,
#       OUTBOUND_FROM (Twilio number, E.164), OUTBOUND_CALL_NUMBER or SIP_INBOUND_ALLOWED_NUMBERS
uv run python -m cli.sip_setup --dry-run && uv run python -m cli.sip_setup
uv run python -m cli.sip_setup --twiml   # Twilio → TwiML Bins → rmsai-inbound; set it as the number's "A call comes in"
make phone-up                            # or: VOICE_WORKER_ROLE=phone uv run python -m cli.voice_worker dev
```

It creates, idempotently by name, on the telephony LiveKit (`voice/sip_setup.py`; `plan()` is pure
and unit-tested, and `--dry-run` prints the exact objects with secrets masked):
- inbound trunk `rmsai-inbound-twilio`: digest auth; `allowed_numbers` = the call-in numbers
  (`SIP_INBOUND_ALLOWED_NUMBERS`, else `OUTBOUND_CALL_NUMBER`) plus `OUTBOUND_FROM` (the caller ID
  on bridged outbound calls);
- dispatch rule `rmsai-inbound-dispatch`: a **callee** rule (room = the SIP user part, exactly, no
  randomization) that **names the agent** `rmsai-agent-phone` and hides the caller's number;
- outbound trunk `rmsai-outbound-twilio` only when `TWILIO_SIP_*` is set (the paid Elastic SIP path).

The rule naming the agent matters: workers use explicit dispatch, so a rule without it would put
callers in a room with nobody in it.

## Inbound (call-in)

The Twilio number's TwiML Bin dials `sip:rmsai-call-{{CallSid}}@<LIVEKIT_SIP_URI>`, giving one room per
call. The trunk accepts only those credentials and allowed callers; the callee rule creates the room
and dispatches the phone agent. The room has no staged alert, so the worker runs the PIN-gated Q&A
handler. **Status:** set up; live test pending.

## Outbound (event alert)

`cli.consume --transport sip` stages the event's alert in the shared Redis for the per-event room
`rmsai-outbound-<event_id>`, then dials and dispatches via `config.telephony()`
(`cli/consume.py: livekit_voice_wiring`).
- **Today** the dial is `create_sip_participant` through an outbound trunk (`LIVEKIT_SIP_TRUNK_ID`),
  which exists only on the paid Elastic SIP path.
- **Next phase (trial-compatible):** the Twilio Calls API rings the clinician; on answer, TwiML
  `<Dial><Sip>sip:rmsai-outbound-<event_id>@…>` bridges into that same room via the callee rule. In
  that mode the relay must **not** also dispatch the agent explicitly, since the rule already does
  and two agents would join.
- Unanswered after retries: with `--notifier twilio` the alert is texted instead (SMS fallback).
- Pass `--number`: it defaults to a placeholder, not `OUTBOUND_CALL_NUMBER`.

## On-demand call (`cli.call`)

`uv run python -m cli.call --caller livekit` rings `OUTBOUND_CALL_NUMBER` without an event, in its
own `rmsai-call-<id>` room, dispatching the agent before the dial (no staged alert ⇒ PIN-gated Q&A).
It needs `LIVEKIT_SIP_TRUNK_ID` (paid path) and `OUTBOUND_FROM`; on a trial account it fails fast
as invalid (exit 2). `--caller simulated` (the default) exercises the path with no telephony.

## Security and data

- **Inbound:** digest credentials plus the caller allow-list, with the worker's PIN gate on top.
  `cli.sip_setup --twiml` prints the password unmasked, since it must go in the TwiML Bin; treat the
  Bin as a secret.
- **Data:** call audio passes through Twilio and LiveKit Cloud. Synthetic or public data only until
  BAAs or a self-hosted SIP stack are in place.

Other fronts (Jambonz, Asterisk or a self-hosted `livekit-sip` with a carrier trunk) would slot in at
the same point, but none is implemented or tested here.
