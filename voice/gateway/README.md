# Voice gateway (phone ↔ LiveKit)

The telephony edge: how a phone call reaches a LiveKit room, where the agent
(`voice/livekit_agent.py`) runs STT → handler → TTS. This file is the engineering summary.
- Full call flows with sequence diagrams: [ARCHITECTURE.md § Telephony split](../../ARCHITECTURE.md#telephony-split--phone-calls-on-a-second-livekit).
- SignalWire setup, step by step: [TELEPHONY_SETUP.md](../../TELEPHONY_SETUP.md).
- Configuration reference: [README § Phone calls](../../README.md#phone-calls-livekit-cloud--signalwire-or-twilio).

## Architecture (telephony split)

A LiveKit SIP bridge only places calls into rooms on its **own** server, and the compose LiveKit
(`infra/livekit.yaml`) has no SIP service. So phone calls run on a second LiveKit, a **LiveKit Cloud**
project (`LIVEKIT_SIP_URL`). The app inbox and WebRTC stay local.
- `Config.telephony()` is the config the call paths use.
- The phone worker registers there as `rmsai-agent-phone` (`VOICE_WORKER_ROLE=phone`,
  `make phone-up`). It pins `LIVEKIT_URL/API_KEY/API_SECRET` before starting livekit-agents' CLI,
  which otherwise reads them from the environment (where `.env` holds the *local* server) and
  registers on the wrong LiveKit.

`TELEPHONY_CARRIER` picks how calls cross between the phone network and LiveKit Cloud:

| | SignalWire (`signalwire`) | Twilio (`twilio`) |
|---|---|---|
| Outbound | LiveKit outbound trunk → SignalWire **Domain App** (TLS, digest) → SWML `connect` (`answer_on_bridge`) → PSTN | trial: none (Calls-API bridge deferred); paid: outbound trunk → Elastic SIP termination |
| Inbound | number's SWML `connect` → `sip:<number>@<LIVEKIT_SIP_URI>;transport=tcp` → inbound trunk (called number + allowed callers) → **individual** rule | TwiML Bin `<Dial><Sip>sip:rmsai-call-{{CallSid}}@…` (digest) → inbound trunk → **callee** rule |
| Carrier-side config | `cli.sip_setup --swml` (two SWML scripts) + a Domain App SIP address | `cli.sip_setup --twiml` (TwiML Bin) on the number |
| Account | must be out of trial mode (trial blocks Domain App traffic) | trial works for inbound; Verified Caller IDs only |

## Provisioning (`cli.sip_setup`)

```bash
uv run python -m cli.sip_setup --dry-run    # masked plan; PROBLEM lines name missing keys
uv run python -m cli.sip_setup              # create/update; prints LIVEKIT_SIP_TRUNK_ID when an outbound trunk exists
uv run python -m cli.sip_setup --swml       # SignalWire scripts  |  --twiml: Twilio TwiML Bin
make phone-up && make phone-logs            # "registered worker" url must be the Cloud URL
```

`voice/sip_setup.py` holds the code: `plan()` is pure and unit-tested; `apply()` runs against the SDK's
`SipService`. Objects are idempotent by name, and each carrier has its own:
- **SignalWire:** `rmsai-outbound-signalwire` (TLS + digest), `rmsai-inbound-signalwire` (matched on
  the hospital's `outbound.from`, `allowed_numbers` = call-in numbers), and `rmsai-inbound-dispatch-signalwire`
  (individual rule, `rmsai-call-…` room per call, agent `rmsai-agent-phone`).
- **Twilio:** `rmsai-inbound-twilio` (digest + `allowed_numbers` = call-in numbers + `outbound.from`)
  and `rmsai-inbound-dispatch` (callee rule: room = the SIP user part, agent named), plus
  `rmsai-outbound-twilio` only when `TWILIO_SIP_*` is set (paid).

Every rule **names the agent**: workers use explicit dispatch, so a rule without it would put callers
in a room with nobody in it.

## Outbound (event alert, on-demand call)

`cli.consume --transport sip`:
1. stages the event's alert in the shared Redis for `rmsai-outbound-<event_id>`;
2. dispatches `rmsai-agent-phone` into that room;
3. dials `CreateSIPParticipant` through `LIVEKIT_SIP_TRUNK_ID` to `--number`, else
   the hospital's `outbound.call_number` (`config/hospitals/<HOSPITAL_ID>.yaml`), with `wait_until_answered` (`cli/consume.py: livekit_voice_wiring`).

`cli.call --caller livekit` is the same dial into `rmsai-call-<id>` with no staged alert, so the
worker runs the PIN-gated Q&A. Any SIP error counts as **no answer**: retried
(`outbound.max_retries`), then the SMS fallback (`--notifier twilio`). A missing trunk or number
fails fast.

## Inbound (call-in)

The carrier forwards the call into LiveKit Cloud (table above). The trunk checks the called number
and/or digest plus the allowed callers; the rule creates a per-call `rmsai-call-…` room and
dispatches the phone agent; with no staged alert the worker runs the PIN-gated Q&A.

## Security and data

- **SignalWire outbound:** the Domain App's credentials authorize calls on your balance; keep them in
  `.env`.
- **SignalWire inbound:** restricted by number and caller allow-list (SWML presents no credentials).
- **Twilio inbound:** digest plus allow-list; the TwiML Bin holds the password.
- **Everywhere:** the worker's PIN gate applies on top.
- **Data:** call audio passes through the carrier and LiveKit Cloud. Synthetic or public data only
  until BAAs or a self-hosted SIP stack are in place.

## Status (2026-10-01)

- **SignalWire:** configured; the first outbound test returned `603 Decline` with no call in
  SignalWire's log (trial mode suspected).
- **Twilio:** call-in set up; the outbound bridge is deferred.
- **SMS fallback:** works (Twilio).
- **Not implemented or tested here:** other fronts (Jambonz, Asterisk, a self-hosted `livekit-sip`).
