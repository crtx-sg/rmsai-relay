# Testing

How to verify each subsystem and the whole loop. Install first: [INSTALL.md](INSTALL.md).

## Testing the speech engines and the knowledge base

Two checklists for "is this subsystem working, and how well?", independent of the companion app.
Commands use `uv run` on the host; inside Docker, swap `uv run python -m` for `$RMSAI`
(`docker compose -f infra/docker-compose.yml run --rm tools python -m`).

### A. Speech engines (TTS and STT)

**What is configured** (`.env`; the voice worker prints a warning at startup only for `stub` or `elevenlabs` backends):

| Setting | Engine | Current demo value |
|---|---|---|
| `STT_BACKEND` | `whisper` (local) / `elevenlabs` (cloud) / `stub` | `elevenlabs` |
| `ELEVENLABS_STT_MODEL` | ElevenLabs Scribe model | `scribe_v2` |
| `WHISPER_MODEL` | faster-whisper model | `base.en` (`tiny.en` is faster, `small.en` more accurate) |
| `STT_LANGUAGE` | pinned language, `auto` to detect | `en` |
| `TTS_BACKEND` | `piper` (local) / `elevenlabs` (cloud) / `stub` | `elevenlabs` |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice | `dkgmw0OA1OcdgloYE1O0` (code default `pNInz6obpgDQGcFmaJgB`, Adam, premade) |
| `ELEVENLABS_TTS_MODEL` | ElevenLabs TTS model | `eleven_flash_v2` |
| `PIPER_VOICE_PATH` | Piper `.onnx` voice file | `~/.local/share/piper/en_US-lessac-medium.onnx` |

```bash
docker logs infra-voice-worker-1 2>&1 | grep -m2 -E 'STT_BACKEND|TTS_BACKEND'   # what the worker runs
```

**1. Round-trip test, no microphone.** `cli.speech_check` synthesizes your text (TTS), transcribes it
back (STT) and compares. Pick each leg independently to compare engines on the same sentence:

```bash
uv run python -m cli.speech_check --tts elevenlabs --stt elevenlabs --text "What were the vitals at the start of this event?"
uv run python -m cli.speech_check --tts elevenlabs --stt whisper    --text "What were the vitals at the start of this event?"
uv run python -m cli.speech_check --tts piper      --stt whisper    --text "Bed four, atrial fibrillation, MEWS five"
uv run python -m cli.speech_check --interactive    # type sentences, 'quit' to exit
uv run python -m cli.speech_check --out /tmp/say.wav   # also save the audio to listen to
```

```
  said  : 'What were the vitals at the start of this event?'
  heard : 'What were the vitals at the start of this event?'  [OK]
  audio : 94252 bytes  |  TTS 743 ms  STT 793 ms
  ...
  said  : 'Bed four, atrial fibrillation, MEWS five'
  heard : 'VED4, atrial fibrillation, NEWS 5.'  [differs]      <- base.en on clinical terms
```

`[differs]` on clean synthetic audio is an engine problem (model too small, vocabulary). Try a
larger `WHISPER_MODEL` or the other backend.

**2. A specific voice or model.** Settings come from `.env`, but a variable set on the command line
wins, so you can try one without editing anything:

```bash
ELEVENLABS_VOICE_ID=<voice_id> uv run python -m cli.speech_check --tts elevenlabs --stt elevenlabs
ELEVENLABS_STT_MODEL=scribe_v1 uv run python -m cli.speech_check --tts elevenlabs --stt elevenlabs
WHISPER_MODEL=small.en         uv run python -m cli.speech_check --tts elevenlabs --stt whisper
PIPER_VOICE_PATH=/path/to/en_GB-alan-medium.onnx uv run python -m cli.speech_check --tts piper --stt whisper
```

ElevenLabs voice ids are listed in the ElevenLabs web app under *Voices*. Limits seen on this
account:

- The **free tier only allows premade voices** via the API. A library voice fails with
  `ElevenLabs API error 402: Free users cannot use library voices via the API`. `pNInz6obpgDQGcFmaJgB`
  (Adam, also the default when `ELEVENLABS_VOICE_ID` is unset) works; `21m00Tcm4TlvDq8ikWAM` (Rachel)
  does not.
- Listing voices through the API (`GET /v1/voices`) needs a key with the `voices_read` permission.

Piper voices are `.onnx` + `.onnx.json` pairs from the Piper voice list (rhasspy/piper-voices); put
both files side by side and point `PIPER_VOICE_PATH` at the `.onnx`.

**3. What the worker heard from the app.** When a spoken question gets a wrong answer, check the
transcript before the KB:

```bash
docker logs -f infra-voice-worker-1 2>&1 | grep -E '\[worker\] (stt|heard|reply|ptt)'
```

```
[worker] ptt: mic open (audio_enabled=True, linked=clinician-355eed)
[worker] stt: transcribing 2.84s of audio (ElevenLabsSTT)
[worker] heard: 'Welcome to the batteries and start of this unit'
[worker] reply: "I don't have information on that in the knowledge base."
```

| Log | Meaning |
|---|---|
| `ptt` start/end but no `stt: transcribing` | no audio reached the worker (mic permission, released too early) |
| `stt: transcribing …` then a garbled `heard:` | audio arrived but was poor: clipped start, low level, noise |
| `heard:` correct, `reply:` wrong | speech is fine; investigate the KB (section B) |

If the round-trip in step 1 is clean but the app transcripts are garbled, the problem is the mic
audio, not the engine: press, wait about half a second, speak, release after you finish; a headset
helps.

> ⚠️ ElevenLabs is a cloud service. TTS text is de-identified first, but **STT sends raw audio**, so
> use synthetic speech only, never real patients (hard rules 4 and 5).

### B. Knowledge base (Neo4j graph + Qdrant vectors)

**1. What is stored.**

```bash
# Neo4j: node counts by type (GUI: http://localhost:7474, neo4j / rmsai_dev_pw)
docker exec infra-neo4j-1 cypher-shell -u neo4j -p rmsai_dev_pw \
  "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"

# Qdrant: chunk count, vector size, and chunks per document (GUI: http://localhost:6333/dashboard)
curl -s localhost:6333/collections/rmsai_docs | python3 -m json.tool | grep -E 'points_count|"size"'
curl -s -X POST localhost:6333/collections/rmsai_docs/points/scroll -H 'content-type: application/json' \
  -d '{"limit":1000,"with_payload":["doc_id"],"with_vector":false}' | python3 -c "
import sys,json,collections
pts=json.load(sys.stdin)['result']['points']
c=collections.Counter(p['payload']['doc_id'].split('#')[0] for p in pts)
[print(f'{n:4d}  {d}') for d,n in c.most_common()]"
```

Expect clinical documents (`afib_rvr.md`, `af_sop.pdf`, …) plus one `report:<event_id>` entry per
consumed event. Vector size 384 means the BGE embedder (`EMBEDDER=bge`); 256 is the hashing embedder.
Every indexing command must use the same embedder as the collection.

**2. One event across both stores.**

```bash
uv run python -m cli.kb_dump --list            # recent event ids
uv run python -m cli.kb_dump <event_id>        # graph node + report file + vector chunks
uv run python -m cli.graph template outstanding_action_items
uv run python -m cli.graph lookup "critical events in the last 24 hours"
```

**3. Add documents.** Markdown in `docs/` (top level) is the built-in corpus; anything else goes
through `cli.kb_upload` (PDF, markdown, text). Uploads are incremental: an unchanged file is a no-op,
an edited one replaces its chunks.

```bash
uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf --dry-run   # preview chunks
uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf             # index it
uv run --extra pdf python -m cli.kb_upload --dir protocols/ --glob '*.pdf' --extract  # + graph entities
uv run python -m cli.kb_vector index --dir docs                                     # re-index docs/
```

Then confirm it is retrievable, and cited by page:

```bash
uv run python -m cli.kb_vector retrieve "rate control in atrial fibrillation"
# [1] score=0.771  (af_sop.pdf#page 1)
# [2] score=0.746  (afib_rvr.md#Overview)
```

Scanned (image-only) PDFs have no text and are rejected rather than indexed empty.

**4. Answer quality for a question.**

```bash
uv run python -m cli.kb_route "what is the SOP for AF?"            # which path: template / LLM router / documents
uv run python -m cli.kb_route --llm --patient PT992591 "how have their vitals been?"
uv run python -m cli.kb --show-context "what is the SOP for atrial fibrillation?"   # answer + evidence blocks
uv run python -m cli.kb --mode vector "what is the SOP for atrial fibrillation?"    # passages only, for comparison
uv run python -m cli.kb_vector ask "what triggers escalation to critical care"      # grounded answer, declines off-corpus
```

Read the result against three questions: is the answer supported by the passages shown, are the
citations the right document and page, and does an off-topic question decline instead of guessing?
Retrieval scores on this corpus with BGE: on-topic 0.61–0.83, off-topic 0.35–0.54; answers need
≥ `KB_MIN_RELEVANCE` (0.60) or a graph relationship. A declined turn logs both numbers and the top
passage in the orchestrator log (see [Managing & inspecting the KB](OPERATIONS.md#managing--inspecting-the-kb)).

**5. Quality and performance benchmark.** `cli.kb_eval` runs a gold question set
(`kb/eval/questions.json`) through vector-only and hybrid retrieval and reports correctness, citation
grounding, context-token cost and latency. Its Qdrant index is in memory, but seeding its own
patient cohort resets the live Neo4j, so without `--no-seed` it refuses unless
`RMSAI_ALLOW_LIVE_GRAPH_RESET=1`:

```bash
uv run python -m cli.kb_eval --no-seed          # safe on a live demo graph
RMSAI_ALLOW_LIVE_GRAPH_RESET=1 uv run python -m cli.kb_eval   # ⚠️ wipes Neo4j; re-run DEMO.md §2 and republish afterwards
uv run python -m cli.kb_eval --no-seed --json   # raw report for comparing runs
```

```
metric                          vector      hybrid
correctness (overall)              71%         71%
  correctness: passage            100%        100%
  correctness: relationship          0%          0%     <- R1/R2 need the eval cohort (no --no-seed)
  correctness: safety             100%        100%
citation grounding                 60%         80%
avg context tokens               260.7       292.6
avg latency                      0.3ms      18.1ms
```

With `--no-seed` the relationship questions miss because they ask about the eval cohort's patients,
which a demo graph doesn't have; passage and safety scores are the meaningful ones there. To check a
new document, add questions about it to a copy of `questions.json` and pass `--questions <file>`.

End-to-end latency of one answer includes model loading and the LLM call: `time uv run python -m
cli.kb "<question>"` takes about 15 s from a cold start on `llama3.2:3b`, most of it process start
and model load. See [8. Performance](OPERATIONS.md#8-performance) for the knobs that change it.

## End-to-end testing

### 0. Demo bring-up sequence

**All in Docker** — one command brings up the stores, LiveKit, and the three app services:

```bash
make docker-up
make docker-ps        # every service should be Up (gateway: Up (healthy))
make docker-logs      # tail consumer + voice-worker + gateway

# drive a scenario: produce an event into the bus (steps 2-4 below run the same way)
docker compose -f infra/docker-compose.yml run --rm tools \
    python -m cli.ingest --file data/inference/<f>.h5 --emit bus
```

**Hybrid** (containers for the infrastructure, host terminals for whatever you're editing) — this
is the loop the rest of this section is written for, since it shows each process's output directly:

```bash
# 1. Backing services only — needed even though the CLIs run on the host
make stores-up
make stores-check          # redis / neo4j / qdrant / livekit reachability

# 2. (voice) agent worker — joins rooms, runs STT → Handler → TTS. Leave running in its own terminal.
uv run python -m cli.voice_worker dev

# 3. Drive a scenario (separate terminals): produce an event, then consume/dial. See steps 2–6 below.
```

Stop the container of any service you want to run by hand — `docker compose -f
infra/docker-compose.yml stop voice-worker` — so the two don't both join the same rooms.

> A voice worker started against a missing LiveKit server just retry-loops on `:7880`
> (`Connect call failed ('127.0.0.1', 7880)`) — that error means *the server isn't running*, not a
> bug in the worker. LiveKit now starts with a plain `up`; it is no longer `later`-profiled.

> If the worker **starts or restarts while the app is connected**, the inbox room can be left
> agent-less for a moment (chat/select hit an empty room → no reply, no speech). The worker
> re-dispatches itself into live `rmsai-inbox-*` rooms on startup (`LIVEKIT_REDISPATCH_ON_START`,
> default on); to re-wire immediately without waiting or re-logging in:
>
> ```bash
> uv run python -m cli.dispatch --all-inbox        # every live inbox room lacking an agent
> uv run python -m cli.dispatch --room rmsai-inbox-h1   # one specific room
> ```
> The agent joins a few seconds later (cold-start loads STT/TTS/orchestrator).

### 1. Tests

```bash
uv run pytest -q -m "not infra"        # offline: no containers needed
uv run pytest -q                       # full suite: needs the stores; graph-wiping tests skip unless RMSAI_ALLOW_LIVE_GRAPH_RESET=1
```

The offline suite passes with no failures; see [§6 of the runbook](OPERATIONS.md#6-test-the-application).

### 2. Generate synthetic events (vendored simulator)

```bash
# writes HDF5 under <cwd>/data/inference/ — run from the repo root, or pass --output-dir
uv run python external/ecgtranscnn/scripts/generate_inference_data.py --output-dir data/inference
```

### 3. Direct relay path (HDF5 → call, bypasses the bus)

```bash
uv run python -m cli.outbound \
    --file data/inference/<file>.h5
```

### 3a. Upload protocols / SOPs / checklists to the KB

Puts your own reference material behind the Q&A, so "what is the SOP for handling a patient with
AF?" is answered from your documents with a citation instead of declined. PDF, markdown, and text.

```bash
uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf --dry-run  # preview only
uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf            # one file
uv run --extra pdf python -m cli.kb_upload --dir protocols/ --glob '*.pdf'        # a folder
uv run --extra pdf python -m cli.kb_upload --dir protocols/ --extract             # + graph entities
```

`--extra pdf` pulls in `pypdf`; drop it if you are only uploading `.md`/`.txt`. `--dry-run` prints
the chunk plan and touches nothing. `--extract` also pulls `Condition`/`Treatment` entities into the
graph, so the hybrid retriever can relate the document to patients and events.

Uploads are idempotent: re-uploading a file replaces all of that document's chunks, so an unchanged
file ends up identical and an edited one leaves no stale sections behind. PDF pages are cited
individually — an answer from `af_sop.pdf#page 3` tells the clinician which page to turn to — and a
scanned/image PDF is rejected with that diagnosis rather than silently indexed as empty.

**The embedder must match the collection.** Vectors are only comparable within one embedder
(hashing=256-dim / BGE=384-dim), so every KB path defaults `--embedder` to **`EMBEDDER`** from
`.env` — set it once and `kb_upload`, `consume`, `outbound`, `text_chat`, and the voice worker all
agree. A mismatch is refused with the fix rather than corrupting the store. (`cli.kb_vector` and `cli.kb`
are the exceptions: they default to `auto`, which prefers BGE and falls back to hashing offline —
pass `--embedder` explicitly there if `EMBEDDER=hashing`; `cli.kb_eval` defaults to `hashing`.)

Each upload is **copied into a managed folder** (`KB_UPLOAD_DIR`, default `data/kb_uploads/`) and a
corpus rebuild re-indexes it automatically, so uploads survive `index --reset`:

```bash
uv run --extra pdf --extra rag python -m cli.kb_vector --embedder bge index --dir docs --reset
# [index] re-indexed 3 uploaded document(s) from data/kb_uploads
# {"indexed_chunks": 18, "embedder": "BAAI/bge-small-en-v1.5", "mode": "reset", "uploads": 3}
```

Pass `--no-keep` to index a file in place without a managed copy (a rebuild will then drop it), or
`--upload-dir ""` on `index` to rebuild from `docs/` alone.

**Sample document to test the path with.** [`docs/samples/critical_alarm_sop.md`](docs/samples/critical_alarm_sop.md)
is a synthetic critical-alarm SOP + escalation checklist (9 sections → 9 chunks). It lives in
`docs/samples/`, which the **non-recursive** `docs/*.md` corpus glob deliberately excludes — so it is
*not* auto-indexed, and uploading it actually exercises the upload path:

```bash
uv run python -m cli.kb_upload --file docs/samples/critical_alarm_sop.md
# [upload] critical_alarm_sop.md: 9 chunk(s) across 9 section(s)
# [upload] indexed critical_alarm_sop.md -> 9 chunk(s) (embedder bge, dim 384)
```

#### Verify the KB

```bash
# ranked chunks + citations — did the document actually land, and does it win the query?
uv run python -m cli.kb_vector --embedder bge retrieve "what is the escalation checklist for a critical alarm" -k 3

# grounded, cited answer (declines when out-of-corpus rather than fabricating)
uv run python -m cli.kb_vector --embedder bge ask "what happens if the on-call clinician does not respond to an alarm"

# hybrid: vector passages + graph relationships, both blocks shown
uv run python -m cli.kb --embedder bge --show-context "which conditions are co-morbid with atrial fibrillation"

# inventory — every document in the collection, by chunk count
curl -s -X POST http://localhost:6333/collections/rmsai_docs/points/scroll \
  -H 'Content-Type: application/json' -d '{"limit":500,"with_payload":true}' \
| python3 -c "import sys,json,collections; c=collections.Counter(p['payload'].get('doc_id') for p in json.load(sys.stdin)['result']['points']); [print(f'{v:4d}  {k}') for k,v in sorted(c.items())]"
```

A healthy result for the sample SOP looks like this — the uploaded document out-ranks the committed
corpus on its own subject, and each hit names the section to cite:

```
[1] score=0.859  (critical_alarm_sop.md#Escalation checklist)
[2] score=0.769  (critical_alarm_sop.md#Escalation ladder and response times)
[3] score=0.756  (critical_alarm_sop.md#Immediate response in the first sixty seconds)
```

The inventory lists `docs/*.md` chunks, each uploaded document, and one `report:<event_id>` group per
consumed event. If an upload is missing from it, the usual causes are: a PDF dropped into `docs/`
(the corpus glob is markdown-only — it must go through `kb_upload`), a file under a subdirectory of
`docs/` (the glob is non-recursive), or `index --reset` run with `--upload-dir ""`. Qdrant's own
dashboard at `http://localhost:6333/dashboard` shows the same collection interactively.

### 3b. On-demand call (no event at all)

Rings the hospital's `outbound.call_number` because you asked it to, not because something happened. Each call gets
its own room (`rmsai-call-<id>`); the agent is dispatched there **before** the dial, and with no
alert staged in that room the worker runs the PIN-gated Q&A handler — so the callee authenticates,
then asks grounded questions. The agent is dispatched again before each redial. `cli.call` is not
tied to an event, so it never sends the SMS fallback; test that with an event (§6).

```bash
uv run python -m cli.call                    # simulated: whole path, no telephony
uv run python -m cli.call --caller livekit   # real SIP via LIVEKIT_SIP_TRUNK_ID (SignalWire)
uv run python -m cli.call --caller livekit --to +15551234567 --no-dispatch   # trunk test only
```

In split mode (`LIVEKIT_SIP_URL` set) the dial and the dispatch go to the telephony LiveKit, so the
phone worker must be running (`make phone-up`). `--caller livekit` needs an outbound trunk
(`LIVEKIT_SIP_TRUNK_ID`) and a caller ID (the hospital's `outbound.from`). `cli.sip_setup` creates the
trunk; without one the call fails fast as invalid (exit 2). See
[Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire). Exit code is 0 answered / 1 no-answer / 2 invalid.

### 4. Bus path: producer → Redis Stream → consumer

```bash
# PRODUCER — classify + publish to rmsai.events
uv run python -m cli.ingest \
    --file data/inference/<file>.h5 \
    --emit bus --stream rmsai.events

# CONSUMER — persist (graph+vector) + criticality-gated dispatch; drain backlog then exit
uv run python -m cli.consume --channel voice --once \
    --follow-up "what were the vitals at the event" --ack "yes I acknowledge"

# text channel instead of voice
uv run python -m cli.consume --channel text --once
```

> Run a **single** consumer per patient — concurrent consumers on the same patient cause Neo4j
> write-lock timeouts (each failure is retried after `--retry-idle-ms`, default 10 min, and after
> `--max-attempts`, default 3, it is moved to `rmsai.events.dead`). Partitioning the bus by
> `patient_id` keeps ordering while patients run in parallel.

### 5. Real WebRTC audio loop (browser, no phone)

```bash
# PREREQ — the LiveKit media server must be running. `make docker-up` (or a plain `up`) starts it;
# skip this if `docker ps` already shows infra-livekit-1 on :7880.
docker compose -f infra/docker-compose.yml up -d livekit

# TERMINAL A — agent worker (joins rooms; runs Whisper STT → Handler → Piper TTS).
# Already running as a container under `make docker-up`; run it by hand only to watch its output,
# and stop the container first so the two don't both join the room:
#   docker compose -f infra/docker-compose.yml stop voice-worker
uv run python -m cli.voice_worker dev
```

**Inbound (ask the KB):**
```bash
uv run python -m cli.livekit_token --room rmsai-call-demo
# → open https://agents-playground.livekit.io → Manual → paste URL + Token → allow mic
# → speak the PIN → ask a KB question
```

**Outbound (event-driven, carries the event context).** Needs the hospital's `outbound.dispatch_mode`
to include `call` (`app+call`, the default). With `app` nothing is staged or called.
```bash
# produce an event (step 4 producer), then stage the alert + print a join token
uv run python -m cli.consume --channel voice --caller livekit --transport webrtc --once
# → join the printed rmsai-outbound-<event_id> room (playground or meet.livekit.io)
# → PIN → hear THIS event's alert → Q&A (see below) → say "acknowledge" (flips event status)
```

> Each event gets its **own** room (`rmsai-outbound-<event_id>`; each redial a fresh one,
> `rmsai-outbound-<event_id>-r<n>`) and a **fresh** join token, both
> printed by `cli.consume` — this holds for **both** transports (SIP and WebRTC), so the room carries
> that event's staged alert. The static `rmsai-outbound` name (`LIVEKIT_SIP_ROOM`) is only the
> *default* for the standalone `cli.outbound` path (one call at a time), not the bus consumer. The
> worker joins when the room is dispatched; a worker started before a code change won't pick it up for
> an already-live room — restart the worker **and** place a new call. With `--transport sip` in split
> mode the room is created on the telephony LiveKit (Cloud), while the staged alert stays in the
> shared Redis, which is how the local phone worker finds it.

**Talking vs typing during a call (modality-matched replies).** Once past the PIN and the spoken
alert, you can interact two ways and the response matches the input modality:

- **Speak** — follow-up *audio* Q&A is gated by a **wake word** (`AUDIO_WAKE_WORD`, default
  `"hey vios"`) so room noise and Whisper hallucinations don't trigger replies. Say e.g.
  *"hey vios, what were the vitals at the time of the event?"* → **spoken** answer. The agent then
  stays "awake" for `AUDIO_WAKE_WINDOW_S` (default 30 s), so immediate follow-ups don't need to
  repeat the wake word. Audio with no wake word (and outside the window) is silently ignored.
  If the wake word keeps mishearing (it's out-of-vocab, so STT mangles it), set
  `AUDIO_WAKE_REQUIRED=false` to answer every authenticated audio turn, and/or route STT to a
  stronger model with `STT_BACKEND=elevenlabs` (synthetic voice only — raw audio leaves the host).
- **Type** — a message in the LiveKit **chat box** gets a **text-only** reply on the chat channel
  (no TTS/audio). Typed turns are never wake-word gated.

The PIN entry and the spoken alert are never wake-word gated, and a reply to the alert (an
**"acknowledge"**, a "yes", a "not yet") passes the gate without the wake word. On an outbound call,
"the event" is the called event (its vitals → `T5`); without an event in scope, patient-scoped
questions use that patient's most recent `MonitoredEvent`.

### 6. Real SIP phone call (outbound to a number)

In Docker: `make demo-phone-up`, then publish events (`cli.ingest … --emit bus`). Every gated
event appears in the app at once and the calls follow one by one, highest criticality first. Use
`make phone-transcript` to watch each conversation. On the host:

```bash
uv run python -m cli.consume --channel voice --caller livekit --transport sip --once \
    --number +1XXXXXXXXXX --notifier signalwire
```

Prerequisites (details in [Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire)):
- The hospital's `outbound.dispatch_mode` includes `call`. `outbound.enabled` does not matter (`cli.consume` forces it on).
- A destination: `--number`, else the hospital's `outbound.call_number` (a real call refuses to start with neither).
- In split mode, the phone worker is running (`make phone-up`), not the step-5A app worker.
- `LIVEKIT_SIP_TRUNK_ID`: the SignalWire outbound trunk (`TELEPHONY_SETUP.md`).
- `--notifier signalwire` texts the alert from the same number if the call goes unanswered: every attempt rang out, was
  busy, or was picked up and hung up with no PIN entered (a PIN lockout is not texted).
- The PIN can be spoken or keyed (4 digits or `#`).
- A call whose LiveKit connection reconnects after the PIN is hung up and rung again (at most 2
  redials, unless the event was acknowledged meanwhile).
- To test the SMS: publish one event (`cli.ingest --file … --emit bus`), then decline every attempt
  or hang up before the PIN. Expect `counted as missed` per pickup, then `SMS fallback: sms_delivered`.
- Restart the consumer or phone worker only when no call is up; mid-call the call stays open.

### Recommended smoke test (clean, single pass)

```bash
uv run python external/ecgtranscnn/scripts/generate_inference_data.py
uv run python -m cli.ingest --file data/inference/<file>.h5 --emit bus
uv run python -m cli.consume --channel voice --once \
    --follow-up "what were the vitals" --ack "yes I acknowledge"
```

Expect (with `dispatch_mode` including `call`; this uses the simulated caller): critical events
(e.g. AFib/High) persisted + called + acknowledged; a confident NORMAL_SINUS persisted but skipped
(`false_positive`), unless the vitals trigger `fp_override (…)`, which is common on simulator data.

### Other CLI harnesses (per-subsystem)

```bash
uv run python -m cli.gen_synthetic ...     # synthetic signal/event generation
uv run python -m cli.kb_vector index --dir docs/   # index the clinical corpus (vector)
uv run python -m cli.kb_upload --dir protocols/    # add SOPs/guidelines/checklists (PDF + markdown)
uv run python -m cli.real_samples pick --pick ...  # held-out real-ECG events from ecg_sigma -> data/real
uv run python -m cli.kb_route --llm "..."  # which path a question takes (regex / LLM / documents)
uv run python -m cli.call --caller livekit # ring the on-call number on demand (no event needed)
uv run python -m cli.inbox_probe --ptt     # probe in-app chat/PTT without the browser
uv run python -m cli.graph migrate         # graph schema: constraints + indexes
uv run python -m cli.graph chain           # rebuild every patient's FOLLOWED_BY event chain (backfill)
uv run python -m cli.kb "..."              # hybrid (vector + graph) KB query
uv run python -m cli.kb_dump <event_id>    # dump one event: graph node + report file + vector chunks
uv run python -m cli.kb_eval               # vector vs hybrid over the gold question set
uv run python -m cli.memory demo           # working + episodic memory round-trip
uv run python -m cli.speech_check          # offline Piper TTS → Whisper STT round-trip
uv run python -m cli.text_chat             # inbound query by text (PIN-gated, KB-grounded)
uv run python -m cli.voice                 # offline typed-text voice demo (no audio hardware)
uv run python -m cli.consume --help        # bus consumer: --channel/--caller/--transport/--notifier/--number
uv run python -m cli.outbound --file <f>.h5  # single-file outbound loop (--no-answer, --fail-delivery, --notifier)
uv run python -m cli.gateway               # companion-app gateway (serves app/, /session, /ack, /events, /event-info, /metrics, /artifact*)
uv run python -m cli.inbox_publish --dry-run   # print a worklist message + token roundtrip (drop --dry-run to push it)
uv run python -m cli.voice_worker dev      # agent worker; VOICE_WORKER_ROLE=phone for the phone worker
uv run python -m cli.livekit_token --room <r>  # join token for a room (playground testing)
uv run python -m cli.dispatch --all-inbox  # re-dispatch the agent into agent-less inbox rooms
uv run python -m cli.sip_setup --dry-run   # telephony trunks + dispatch rule (--swml prints the SignalWire scripts)
uv run python -m cli.model_perf --events   # model performance from the graph (TP/FP/FN/TN, CIs, per class, why)
uv run python -m cli.events --status pending   # every stored event, filtered; --ask "<q>" shows what the agent would say
```
