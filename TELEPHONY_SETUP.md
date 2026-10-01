# Telephony setup — LiveKit Cloud + SignalWire

Step-by-step guide to real phone calls: the relay calls a clinician's phone about a critical event,
and a clinician calls in to ask about events. Everything else (the app, WebRTC calls, local demos)
needs none of this; see [README § Local-only mode](README.md#local-only-mode-no-livekit-cloud-no-phone-carrier).

How it works and why: [README § Phone calls](README.md#phone-calls-livekit-cloud--signalwire-or-twilio)
and the call-flow diagrams in [ARCHITECTURE.md](ARCHITECTURE.md#telephony-split--phone-calls-on-a-second-livekit).
The Twilio alternative is covered there too; this guide is SignalWire only.

> **Status (2026-10-01):** everything below is configured on the dev accounts. The first outbound
> test returned `603 Decline` from SignalWire with no call in SignalWire's log; the leading suspect
> is trial mode (see [Troubleshooting](#troubleshooting)). Call-in has not been tested yet.

---

## What you are building

```
OUTBOUND (event alert / on-demand call)
relay ──CreateSIPParticipant──► LiveKit Cloud ──SIP INVITE (TLS, digest)──► SignalWire Domain App
                                  (room rmsai-outbound-<event_id>                 │ SWML "rmsai-outbound":
                                   or rmsai-call-<id>; agent already in it)       │ connect → PSTN
                                                                                  ▼
                                                                        clinician's phone rings

INBOUND (call-in)
clinician's phone ──► SignalWire number ──SWML "rmsai-inbound": connect──► LiveKit Cloud SIP URI
                                                           (TCP)               │ inbound trunk (your number,
                                                                               │ allowed callers) → dispatch
                                                                               ▼ rule → rmsai-call-… room
                                                                     agent rmsai-agent-phone: PIN → Q&A
```

The agent (`rmsai-agent-phone`) runs **on your machine**: STT, TTS, LLM and the knowledge base stay
local. It connects *out* to LiveKit Cloud, so no ports need opening, and WSL2 is fine.

## Prerequisites

| Need | Notes |
|---|---|
| The local stack runs | `make docker-up` works; see the README's Setup |
| A **LiveKit Cloud** project | free tier is fine for development |
| A **SignalWire** account **out of trial mode** | trial mode blocks Domain App traffic and international calls. Leave it by adding a card and funding ≥ $5 (a promotional credit may not count) |
| A phone to test with | the destination; for an international number (e.g. `+91…`) check that SignalWire permits that country |

---

## 1. LiveKit Cloud project

1. Create a project (e.g. **VoiceAppTest**, region nearest you).
2. **Settings → Keys**: create an API key. Note the **URL** (`wss://<project>.livekit.cloud`), the
   **API key** and the **secret**.
3. **Settings → SIP URI**: note the host, e.g. `1cxgu99y5r5.sip.livekit.cloud` (without `sip:`).

`.env`:

```bash
LIVEKIT_SIP_URL=wss://<project>.livekit.cloud
LIVEKIT_SIP_API_KEY=<key>
LIVEKIT_SIP_API_SECRET=<secret>
LIVEKIT_SIP_URI=<id>.sip.livekit.cloud
```

Keep `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` pointing at your **local** LiveKit. Do
not put the Cloud values there: that would move the app inbox and WebRTC to Cloud as well.

## 2. SignalWire account basics

1. **API** (dashboard → API): note the **Space URL** (`<space>.signalwire.com`), **Project ID** and an
   **API token**. The relay uses these for read-only checks; the call path itself doesn't.
2. **Phone Numbers → Buy**: one voice-capable number. It becomes `OUTBOUND_FROM` (the caller ID and
   the trunk's number).

`.env`:

```bash
TELEPHONY_CARRIER=signalwire
SIGNALWIRE_SPACE=<space>.signalwire.com
SIGNALWIRE_PROJECT_ID=<project id>
SIGNALWIRE_API_TOKEN=<token>
OUTBOUND_FROM=+1XXXXXXXXXX            # the SignalWire number
OUTBOUND_CALL_NUMBER=+XXXXXXXXXXX     # the clinician / your mobile (also the default allowed caller)
```

## 3. The two SWML scripts

Generate them from your config:

```bash
uv run python -m cli.sip_setup --swml
```

It prints two scripts with your values filled in:

```yaml
# outbound: LiveKit → PSTN (runs on the Domain App)
version: 1.0.0
sections:
  main:
    - connect:
        answer_on_bridge: true
        from: "+1XXXXXXXXXX"
        to: "%{call.to.replace(/^sip:/i, '').replace(/@.*/, '')}"

# inbound: your number → LiveKit
version: 1.0.0
sections:
  main:
    - connect:
        to: "sip:%{call.to}@<id>.sip.livekit.cloud;transport=tcp"
```

- **outbound:** `to` strips the SIP URI LiveKit dialled down to the bare phone number.
  `answer_on_bridge: true` keeps the LiveKit leg ringing until the phone actually answers, which is
  what makes "answered" mean answered on the relay side.
- **inbound:** forwards the call to LiveKit Cloud, keeping the dialled number (`call.to`) as the SIP
  user, so LiveKit's inbound trunk can match it.

In the dashboard: **Resources → Add New → SWML Script**, twice: `rmsai-outbound` and `rmsai-inbound`.

## 4. The Domain App (outbound entry point)

LiveKit sends its outbound calls to a **SIP domain** that runs the outbound script.

1. Open the `rmsai-outbound` script → **Addresses** → **Add → SIP Address**.
2. Choose a name (e.g. `rmsairelay`), a **username** (e.g. `rmsai`) and a **password**.
3. Note the resulting **SIP domain**, e.g. `<space>-public.dapp.signalwire.com`. Check that its call
   handler is the `rmsai-outbound` script.

`.env`:

```bash
SIGNALWIRE_SIP_DOMAIN=<space>-public.dapp.signalwire.com
SIGNALWIRE_SIP_USERNAME=rmsai
SIGNALWIRE_SIP_PASSWORD=<password>
```

Anyone with these credentials can place calls on your balance through this script: keep them in
`.env` only.

## 5. Route the number to LiveKit (inbound)

**Phone Numbers → your number → Edit Settings → Assign Resource → `rmsai-inbound`**, then save.

## 6. Create the LiveKit side

```bash
uv run python -m cli.sip_setup --dry-run   # the plan; any PROBLEM line names a missing key
uv run python -m cli.sip_setup             # create/update (idempotent, by name)
```

This creates, in the LiveKit Cloud project:

| Object | Name | What it does |
|---|---|---|
| outbound trunk | `rmsai-outbound-signalwire` | address = `SIGNALWIRE_SIP_DOMAIN`, transport **TLS**, digest credentials, number = `OUTBOUND_FROM` |
| inbound trunk | `rmsai-inbound-signalwire` | accepts calls *to* `OUTBOUND_FROM`, only *from* the allowed callers (`SIP_INBOUND_ALLOWED_NUMBERS`, else `OUTBOUND_CALL_NUMBER`). Never created with an empty allow-list |
| dispatch rule | `rmsai-inbound-dispatch-signalwire` | one `rmsai-call-…` room per inbound call; dispatches `rmsai-agent-phone` |

It ends with `LIVEKIT_SIP_TRUNK_ID=ST_…`. Paste that line into `.env`. **The dry run can't print it,
because the id doesn't exist until the trunk is created.** Re-running updates the same objects and
prints the same id.

## 7. Start the phone worker

```bash
make phone-up
make phone-logs     # expect both lines below
```

```
[worker] role=phone server=wss://<project>.livekit.cloud agent=rmsai-agent-phone health_port=8082
{"message": "registered worker", "agent_name": "rmsai-agent-phone", "url": "wss://<project>.livekit.cloud", …}
```

The `url` in **"registered worker"** must be the Cloud URL. If it says `ws://localhost:7880`, the worker
registered on the local LiveKit and no phone call can reach it (fixed in `fe89028`; update and
restart).

## 8. Test

```bash
# A. On-demand outbound: rings OUTBOUND_CALL_NUMBER, the agent asks for the PIN
uv run python -m cli.call --caller livekit

# B. Inbound: call your SignalWire number from an allowed phone → PIN prompt → ask a question

# C. Event-driven alert call (the consumer container on the SIP transport)
#    .env: DISPATCH_MODE=app+call ; then (Compose reads this from the shell, not .env):
CONSUME_ARGS="--channel voice --caller livekit --transport sip --number +XXXXXXXXXXX" make docker-up
#    publish an event (DEMO.md §5); the phone rings → PIN → spoken alert → Q&A → "acknowledge"
```

`cli.call` exits 0 when answered, 1 when not answered after retries, 2 when the setup is invalid.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SIP dial failed … sip status: 603: Decline`, and **no call in SignalWire's voice log** | SignalWire refused the INVITE before running the script: trial mode (Domain App traffic blocked), wrong SIP credentials, or a blocked destination country | 1) Dashboard: no trial banner; add a card + fund. 2) Re-set the SIP address password, update `SIGNALWIRE_SIP_PASSWORD`, re-run `cli.sip_setup`. 3) Permit the destination country (international) |
| A call **in** SignalWire's log that failed after starting | the script ran but `connect` failed (destination, caller ID, balance) | read the log entry's error in the SignalWire dashboard |
| `registered worker … "url": "ws://localhost:7880"` for `rmsai-agent-phone` | phone worker on the wrong server | update past `fe89028`, `make phone-down phone-up` |
| `VOICE_WORKER_ROLE=phone needs LIVEKIT_SIP_URL` | split mode not configured | step 1 |
| `[sip_setup] PROBLEM: SIGNALWIRE_SIP_… is empty` | step 4 not in `.env` | step 4 |
| `LIVEKIT_SIP_TRUNK_ID is not set` / `outcome=invalid` | step 6's id not pasted | step 6 |
| `cli.call` answered but silence | the phone worker isn't running, or isn't on Cloud | step 7 |
| Calling in rings out / is rejected | the number has no `rmsai-inbound` resource, or the caller isn't allowed | step 5; `SIP_INBOUND_ALLOWED_NUMBERS` |
| `no destination: pass --number or set OUTBOUND_CALL_NUMBER` | a real call with no destination | set `OUTBOUND_CALL_NUMBER` or pass `--number` |

**Note on retries:** a SIP error such as `603` counts as "no answer", so the relay retries
(`OUTBOUND_MAX_RETRIES`, every `OUTBOUND_RETRY_DELAY_S`) and then, with `--notifier twilio`, sends
the SMS fallback, even when the cause is configuration. Fix configuration errors before running
event-driven calls.

**Reading SignalWire's side:** dashboard → **Logs → Voice** shows every call SignalWire accepted. An
empty log after a `603` means the INVITE was refused at the door.

## Security and data

- **Outbound:** the Domain App's SIP credentials authorize calls on your balance. Keep them in `.env`
  (gitignored), never in notes or chat.
- **Inbound:** SWML `connect` presents no SIP credentials, so the LiveKit inbound trunk is restricted
  by the called number and the allowed-caller list. The agent's PIN gate applies on top.
- **Data:** call audio passes through SignalWire and LiveKit Cloud (and ElevenLabs, if it is the
  STT/TTS backend). Synthetic or public data only until BAAs or self-hosted SIP are in place
  (CLAUDE.md rules 4/5).
- **SMS:** the SMS fallback currently sends through Twilio (`--notifier twilio`); SignalWire messaging
  is not wired in.

## Switching carriers

`TELEPHONY_CARRIER=twilio|signalwire` selects which objects `cli.sip_setup` creates. Each carrier's
objects have their own names, so both can exist in one LiveKit project. Only the trunk id in
`LIVEKIT_SIP_TRUNK_ID` decides which carrier outbound calls use.
