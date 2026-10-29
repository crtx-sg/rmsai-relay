# Telephony setup — LiveKit Cloud + SignalWire

Step-by-step guide to real phone calls: the relay calls a clinician's phone about a critical event,
and a clinician calls in to ask about events. Everything else (the app, WebRTC calls, local demos)
needs none of this; see [DEMO.md § Local-only mode](DEMO.md#local-only-mode-no-livekit-cloud-no-phone-carrier).

How it works and why: [§ Phone calls](#phone-calls-livekit-cloud--signalwire) below
and the call-flow diagrams in [ARCHITECTURE.md](ARCHITECTURE.md#telephony-split--phone-calls-on-a-second-livekit).

> **Status (2026-10-13):** verified end to end on the dev accounts: outbound alert calls (PIN →
> alert → Q&A → acknowledge), inbound call-in, and the SignalWire SMS fallback (a test SMS to a +91
> number was delivered). The account had to leave trial mode, and `SIGNALWIRE_SIP_USER` must be set
> (step 4), or SignalWire answers `603 Decline`.
> **2026-10-29:** a pickup with no PIN entered counts as a missed call (retried, then texted);
> each redial gets its own room; the PIN can be keyed as well as spoken.

---

## What you are building

```
OUTBOUND (event alert / on-demand call)
relay ──CreateSIPParticipant──► LiveKit Cloud ──SIP INVITE (TLS, digest)──► SignalWire Domain App
                                  (room rmsai-outbound-<event_id>[-r<n>]          │ SWML "rmsai-outbound":
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
| The local stack runs | `make docker-up` works; see [INSTALL.md](INSTALL.md) |
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
   **API token**. The relay uses these to send the SMS fallback (`--notifier signalwire`) and for read-only checks;
the SIP call path itself doesn't.
2. **Phone Numbers → Buy**: one voice-capable number. It becomes the hospital's `outbound.from` (the
   caller ID and the trunk's number).

`.env`:

```bash
TELEPHONY_CARRIER=signalwire
SIGNALWIRE_SPACE=<space>.signalwire.com
SIGNALWIRE_PROJECT_ID=<project id>
SIGNALWIRE_API_TOKEN=<token>
```

The phone numbers are per hospital, in `config/hospitals/<HOSPITAL_ID>.yaml` (copy
`h1.example.yaml`; the file is gitignored). Quote them:

```yaml
outbound:
  from: "+1XXXXXXXXXX"            # the SignalWire number
  call_number: "+XXXXXXXXXXX"     # the clinician / your mobile (also the default allowed caller)
  inbound_allowed_numbers: []     # other numbers allowed to call in
```

`OUTBOUND_FROM` / `OUTBOUND_CALL_NUMBER` / `SIP_INBOUND_ALLOWED_NUMBERS` in `.env` still work, as
logged overrides.

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
        to: "%{call.headers.find(h => h.name.toLowerCase() == 'x-rmsai-to').value}"

# inbound: your number → LiveKit
version: 1.0.0
sections:
  main:
    - connect:
        to: "sip:%{call.to}@<id>.sip.livekit.cloud;transport=tcp"
```

- **outbound:** with `SIGNALWIRE_SIP_USER` set (step 4; required, since a SignalWire SIP address
  routes only its own user part), LiveKit dials `sip:<user>@<domain>` and sends the callee in an
  `X-RMSAI-To` header, which `to` reads. Without it the script uses `call.to`, which SignalWire
  declines with 603. Set step 4's values, then re-run `--swml`.
  `answer_on_bridge: true` keeps the LiveKit leg ringing until the phone actually answers. The relay
  then counts a connected call as answered only if the PIN was accepted or locked out (see *Call flow*, step 5).
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
SIGNALWIRE_SIP_USER=rmsai          # the SIP address's user part (sip:rmsai@…); the callee rides in X-RMSAI-To
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
| outbound trunk | `rmsai-outbound-signalwire` | address = `SIGNALWIRE_SIP_DOMAIN`, transport **TLS**, digest credentials, number = `outbound.from` |
| inbound trunk | `rmsai-inbound-signalwire` | accepts calls *to* `outbound.from`, only *from* the allowed callers (`outbound.inbound_allowed_numbers`, else `outbound.call_number`). Never created with an empty allow-list |
| dispatch rule | `rmsai-inbound-dispatch-signalwire` | one `rmsai-call-…` room per inbound call; dispatches `rmsai-agent-phone` |

It prints `LIVEKIT_SIP_TRUNK_ID=ST_…` (under "Set in .env:", before the SignalWire checklist). Paste that line into `.env`. **The dry run can't print it,
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
# A. On-demand outbound: rings the hospital's outbound.call_number, the agent asks for the PIN
#    (say it, or key 4 digits / #). It redials on no answer but never texts: it has no event.
uv run python -m cli.call --caller livekit

# B. Inbound: call your SignalWire number from an allowed phone → PIN prompt → ask a question

# C. Event-driven alert call (the consumer container on the SIP transport)
#    hospital file: outbound.dispatch_mode: app+call (the default); then the consumer on SIP + the phone worker:
make demo-phone-up        # = CONSUME_ARGS="--channel voice --caller livekit --transport sip --notifier signalwire" + --profile telephony
#    publish an event (DEMO.md §5); the phone rings → PIN → spoken alert → Q&A → "acknowledge"
```

`cli.call` exits 0 when answered, 1 when not answered after retries, 2 when the setup is invalid.
To test the SMS fallback, use C: publish one event (`cli.ingest --file … --emit bus`) and decline
every attempt, or pick up and hang up without the PIN.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SIP dial failed … sip status: 603: Decline`, and **no call in SignalWire's voice log** | SignalWire refused the INVITE before running the script: `SIGNALWIRE_SIP_USER` unset (LiveKit dialled `sip:<number>@domain`), trial mode (Domain App traffic blocked), wrong SIP credentials, or a blocked destination country | 0) Set `SIGNALWIRE_SIP_USER` (step 4) and re-paste `--swml`'s outbound script. 1) Dashboard: no trial banner; add a card + fund. 2) Re-set the SIP address password, update `SIGNALWIRE_SIP_PASSWORD`, re-run `cli.sip_setup`. 3) Permit the destination country (international) |
| A call **in** SignalWire's log that failed after starting | the script ran but `connect` failed (destination, caller ID, balance) | read the log entry's error in the SignalWire dashboard |
| `registered worker … "url": "ws://localhost:7880"` for `rmsai-agent-phone` | phone worker on the wrong server | update past `fe89028`, `make phone-down phone-up` |
| `VOICE_WORKER_ROLE=phone needs LIVEKIT_SIP_URL` | split mode not configured | step 1 |
| `[sip_setup] PROBLEM: SIGNALWIRE_SIP_… is empty` | step 4 not in `.env` | step 4 |
| `LIVEKIT_SIP_TRUNK_ID is not set` / `outcome=invalid` | step 6's id not pasted | step 6 |
| `cli.call` answered but silence | the phone worker isn't running, or isn't on Cloud | step 7 |
| Calling in rings out / is rejected | the number has no `rmsai-inbound` resource, or the caller isn't allowed | step 5; `outbound.inbound_allowed_numbers` in the hospital file |
| `no destination: pass --number or set OUTBOUND_CALL_NUMBER …` | a real call with no destination | set `outbound.call_number` in the hospital file, or pass `--number` (`--to` for `cli.call`) |
| Events alert in the app but **no phone rings**; the consumer log shows `-> answered (attempts 1)` plus a `[webrtc] join room to take the call:` token | the consumer runs `--transport webrtc` (the `make demo-up` default; the "answer" is simulated) | `make demo-phone-up` |
| The call connects but the caller hears **silence**; the phone worker logs `ElevenLabs API error 401 … quota` | the ElevenLabs credits ran out (TTS and STT both fail) | top up, or switch `TTS_BACKEND`/`STT_BACKEND` to piper/whisper |
| `--notifier signalwire needs …` | SMS keys missing | set the keys it names in `.env`, and `outbound.from` in the hospital file |
| Phone picked up, yet the log says `ended with no PIN entered: counted as missed` | a decline after pickup, or the carrier's voicemail | expected: it rings again, then texts (`SMS fallback: sms_delivered`) |
| A retry is answered, the PIN prompt plays, then nothing you say registers | the caller's audio never reached the worker: no `[worker] receiving audio from clinician (SIP)` line in `make phone-logs` | key the PIN instead; report the time and logs |
| The call hangs up mid-Q&A and the phone rings again | `[worker] LiveKit reconnected mid-call`: after a reconnect our audio no longer reaches the phone | expected; at most 2 redials, only if the PIN had been accepted |
| After 3 wrong PINs: no more calls, no SMS | the lockout (`rmsai:calls:locked:<event_id>`, 24 h) | acknowledge in the app, or call in; the next call starts with 3 fresh tries |
| First dial after a consumer restart fails with `Couldn't build proto file into descriptor pool` | a consumer older than `a4c7dee` | update and restart (the protobufs now load at startup) |
| The call stays open after a restart and the agent answers as a KB line | the consumer or phone worker was restarted during the call | hang up; restart only when no call is up |
| A call went unanswered but no SMS arrived | a pickup with no PIN is a missed attempt (`ended with no PIN entered: counted as missed`); after the last attempt the consumer log says which: `[sms] SIMULATED, not sent` (consumer runs without `--notifier`), `rejected (…)` (SignalWire's code), or `sent via signalwire` | `make demo-phone-up`; or check SignalWire's **Logs → Messages** (carrier filtering abroad) |

**Note on retries:** a SIP error such as `603` counts as "no answer", so the relay retries
(`OUTBOUND_MAX_RETRIES`, every `OUTBOUND_RETRY_DELAY_S`) and then, with `--notifier signalwire`, sends
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
  STT/TTS backend), and the SMS fallback sends the pseudonymized alert text through SignalWire.
  Synthetic or public data only until BAAs or self-hosted SIP are in place (CLAUDE.md rules 4/5).
- **SMS:** the SMS fallback sends through SignalWire's Messages API (`--notifier signalwire`, the
  `make demo-phone-up` default) from the same number, which must be SMS-capable (Phone Numbers →
  your number). Sent texts appear under **Logs → Messages**.

---

## Phone calls (LiveKit Cloud + SignalWire)

**Two LiveKits.** A LiveKit SIP bridge can only connect calls into rooms on its own server, and the
local LiveKit has no SIP service. So phone calls run on a second LiveKit, a **LiveKit Cloud** project
(`LIVEKIT_SIP_URL`), while the app inbox, chat and WebRTC stay on the local one. `Config.telephony()`
(`common/config.py`) is the config the call paths use: in split mode it swaps only the LiveKit
URL/key/secret, the agent name (`rmsai-agent-phone`), the worker health port (`8082`), clears the public URL, and turns off
inbox auto-redispatch. Redis (staged alerts), the stores and the numbers are shared. With
`LIVEKIT_SIP_URL` unset everything behaves as a single server, exactly as before.

**The carrier is SignalWire**, a native SIP trunk both ways (`TELEPHONY_CARRIER=signalwire`, the
default and only value):

| | SignalWire |
|---|---|
| Outbound (relay → phone) | LiveKit outbound trunk → SignalWire Domain App → SWML `connect` → PSTN |
| Inbound (phone → agent) | number's SWML `connect` → LiveKit inbound trunk (matched on the number + allowed callers) → individual rule |
| SMS | SignalWire Messages API, from the same number |
| Account requirement | **out of trial mode** (trial blocks Domain App traffic) |
| Carrier-side config | two SWML scripts (`cli.sip_setup --swml`) + a Domain App SIP address |

### Call flow (technical)

What happens on the wire. Diagrams: [ARCHITECTURE.md](ARCHITECTURE.md#telephony-split--phone-calls-on-a-second-livekit).

**Outbound alert (`cli.consume --transport sip`; the same dial for `cli.call --caller livekit`):**
0. The consumer gates the event (`should_call`), pushes it to the app worklist straight away, and
   **queues the call** (Redis `rmsai:calls:*`, highest criticality first, then first-come). One
   dispatcher places queued calls one at a time, each waiting for the previous hang-up. A queued
   call is skipped if the event was acknowledged (e.g. in the app) while it waited, and an unfinished
   call is re-queued after a consumer restart. Restart the consumer or phone worker only when no call
   is up: mid-call, the worker dies before it hangs up and the phone stays connected. Log lines: `[queued] … call N in queue`, `[calls]
   dialling …` / `skip …`, `[call] … -> answered`.
1. At dial time it **stages the spoken alert in Redis** under the room name
   `rmsai-outbound-<event_id>` (TTL 15 min); each redial uses a fresh room,
   `rmsai-outbound-<event_id>-r<n>`, with the alert staged there again. `cli.call` uses `rmsai-call-<id>` and stages nothing.
2. It **dispatches** `rmsai-agent-phone` into that room on the telephony LiveKit, in parallel with
   the dial (the agent joins in 1-4 s while the phone rings), so it is there when the phone answers.
3. It **dials**: `CreateSIPParticipant` on the telephony LiveKit with `LIVEKIT_SIP_TRUNK_ID`, the
   number (`--number`, else the hospital's `outbound.call_number`), caller ID `outbound.from`,
   `ringing_timeout=SIP_RINGING_TIMEOUT_S`, `max_call_duration=SIP_MAX_CALL_DURATION_S` and
   `wait_until_answered`.
4. LiveKit Cloud sends the SIP INVITE through the outbound trunk, over TLS with digest credentials
   to the Domain App, whose SWML `connect` (`answer_on_bridge`) rings the PSTN number and answers
   the LiveKit leg only when the phone does.
5. **Answered:** the phone joins the room as a participant. The worker finds the staged alert
   (`OutboundHandler`): PIN, then the spoken alert, then grounded Q&A (wake word), then
   "acknowledge" sets the event `acknowledged`. "This event" in a question means the called event.
   The PIN can be spoken or keyed (4 digits or `#`; `*` starts over); 3 wrong PINs lock that call out
   (the count is per call). The consumer waits for the hang-up, then reads the worker's flags
   (`rmsai:calls:answered:<event_id>` / `rmsai:calls:locked:<event_id>`, 24 h TTL). Neither set (a
   decline after pickup, or voicemail) logs `ended with no PIN entered: counted as missed` and goes to
   step 6. A lockout is not retried and not texted.
6. **Not answered, or a SIP error** (any failure, including `408` or `603`, counts as no answer):
   the relay stages the alert in a fresh room, dispatches the agent there and retries `OUTBOUND_MAX_RETRIES` times,
   `OUTBOUND_RETRY_DELAY_S` apart. Then, with `--notifier signalwire` (the `make demo-phone-up`
   default), it texts the alert (status stays `reported` if delivered, else `notify_failed`; an event acknowledged
meanwhile is not texted and stays `acknowledged`). With
   the default `simulated` nothing is sent: the log says `[sms] SIMULATED, not sent` and the event
   records `sms_simulated` / `notify_failed`. An invalid or unset trunk or number fails fast with
   no retry.
7. **Reconnect mid-call:** after a LiveKit signal reconnect our audio no longer reaches the phone, so
   the worker hangs up. If the PIN had been accepted it flags the event dropped and the consumer rings
   it again (at most 2 times, unless it was acknowledged meanwhile; log `dropped on a network
   reconnect; ringing again`). A reconnect before the PIN only ends that attempt, and an unanswered
   round never starts a new one after the SMS.

**Inbound call-in:**
1. The SignalWire number's SWML `connect`s to `sip:<number>@<LIVEKIT_SIP_URI>;transport=tcp`.
2. The inbound trunk matches the called number and accepts only allowed callers; the
   **individual** rule creates a `rmsai-call-…` room per call and dispatches `rmsai-agent-phone`.
3. The room has no staged alert, so the worker runs the **PIN-gated Q&A** handler.

### Configure (`.env`)

| Key | What |
|---|---|
| `LIVEKIT_SIP_URL` / `LIVEKIT_SIP_API_KEY` / `LIVEKIT_SIP_API_SECRET` | the LiveKit Cloud project (Settings → Keys) |
| `LIVEKIT_SIP_URI` | its SIP host (Settings → SIP URI), without `sip:`, e.g. `abc123.sip.livekit.cloud` |
| `outbound.from` *(hospital file)* | the SignalWire number (E.164): caller ID, trunk number, and the SMS sender |
| `outbound.call_number` / `outbound.inbound_allowed_numbers` *(hospital file)* | default call destination; who may call in (the explicit list, else the on-call number, else nobody) |
| `LIVEKIT_SIP_TRUNK_ID` | the outbound trunk's `ST_…` id, printed by `cli.sip_setup` (not by `--dry-run`) |
| `LIVEKIT_SIP_AGENT_NAME` / `LIVEKIT_SIP_WORKER_HTTP_PORT` | phone worker identity; defaults `rmsai-agent-phone` / `8082` |
| `SIGNALWIRE_SPACE` / `_PROJECT_ID` / `_API_TOKEN` | the SignalWire API: SMS (`--notifier signalwire`) and read-only checks |
| `SIGNALWIRE_SIP_DOMAIN` / `_SIP_USER` / `_SIP_USERNAME` / `_SIP_PASSWORD` | the outbound script's Domain App SIP address, its user part, and its credentials |

### SMS fallback

`--notifier simulated|signalwire` on `cli.consume` and `cli.outbound` selects the SMS backend for
the text channel **and** for voice calls that go unanswered after retries.

**When it fires:** the call is not answered after `1 + max_retries` attempts (ring timeout, busy,
any SIP error such as `408` or `603`, or a pickup with no PIN entered, voicemail included). A call
answered with the right PIN sends no SMS, and neither does a PIN lockout. `cli.call` never texts (it
has no event). Nor does an event acknowledged elsewhere (e.g. in the app) while the call rang; it stays
acknowledged. An invalid number or missing trunk fails fast and is never texted.

**What it sends:** "Missed call from RMS relay. <spoken alert>" to the call's destination
(`--number`, else the hospital's `outbound.call_number`), from the hospital's `outbound.from`.

| Result | `delivered_sms` | Event status |
|---|---|---|
| sent | `sms_delivered` | `reported` (alerted, awaiting an ack) |
| rejected / network error | `sms_failed` | `notify_failed` |
| `--notifier simulated` (nothing sent; log: `[sms] SIMULATED, not sent`) | `sms_simulated` | `notify_failed` |

Each attempt is audited as `sms_fallback`. The SignalWire sender uses its REST Messages API over
stdlib HTTP (no SDK) and needs `SIGNALWIRE_SPACE`, `SIGNALWIRE_PROJECT_ID`, `SIGNALWIRE_API_TOKEN`
and an SMS-capable `outbound.from`. A send logs `[sms] sent via signalwire to +9198…1234 (message
<sid>)`; a rejection returns "not delivered" with SignalWire's code logged and the number masked,
and never crashes the event. `make demo-phone-up` uses it by default
(`DEMO_PHONE_NOTIFIER=simulated` to send nothing). Each message costs about $0.18 per segment to
India; an alert text is typically 2 segments.

**International delivery:** texts from a US number to non-US mobiles may be filtered by the
destination's carriers; SignalWire's **Logs → Messages** shows the delivery status.
