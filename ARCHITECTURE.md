# Architecture — control flow, data flow & models

A plain-language map of how `rmsai-relay` works end-to-end: what runs when the ECG model is invoked,
how the LLM answers questions, how the voice pipeline speaks, and which model sits where. For the
phase plan, contracts, and config keys see [README.md](README.md).

## The big picture: two pipelines

The system is **two separate flows** that meet at the databases:

1. **Event pipeline** (machine → alert): a device signal arrives, the ECG model classifies it, it's
   persisted, and *if it's clinically significant* a call is placed. **No human involved yet.**
2. **Conversation pipeline** (human ↔ knowledge): a clinician on a phone/browser asks questions;
   answers are grounded in exactly what the event pipeline saved.

The two never call each other directly — they hand off through **Neo4j** (graph) and **Qdrant**
(vector). The event pipeline *writes* them; the conversation pipeline *reads* them.

**Where Redis fits (partial decoupling).** The *knowledge* hand-off is via the databases above; Redis
adds **asynchronous / cross-process** decoupling at two specific seams:
- **Event bus** (Redis Stream `rmsai.events`) — *within* the event pipeline: the producer
  (`cli.ingest --emit bus`, `XADD`) is fully decoupled from the consumer (`bus_consumer`, which
  drains, persists, and dispatches). Producer and consumer run independently.
- **OutboundAlert store** (Redis key `rmsai:outbound_alert:{sid}`, TTL) — *between* the pipelines:
  the consumer that places a call writes the "who/why we're calling" context; the **voice worker is a
  separate process** and reads it on joining the LiveKit room.

Everything else is synchronous: a single Q&A turn calls the orchestrator **in-process**, and the deep
knowledge lives in Neo4j/Qdrant — so the pipelines are **partially** decoupled via Redis, not fully.

```
  EVENT PIPELINE  (device → alert)                    CONVERSATION PIPELINE (clinician Q&A)
  ─────────────────────────────────                   ──────────────────────────────────────
  HDF5 file / MQTT                                     caller audio ─► STT ─► text
        │  ingest/                                            │ (Whisper or ElevenLabs)
        ▼                                                     ▼  VAD (Silero) marks end-of-speech
  SignalWindow (ECG + vitals + history)                PIN gate (no PHI until authenticated)
        │                                                     │
        ▼  inference/                                         ▼
  ECG model .predict() ─► (event_type, confidence)     Handler  ("the LLM node", but really…)
        +  Vitals → MEWS score + trends                       │
        +  FP gate / criticality                              ▼
        ▼                                              Orchestrator (one turn):
  DeviceEvent (+care guidance +markdown report)          guardrails → retrieve (RAG) → de-id
        │                                                  → LLM generate → guardrails
        ├─► Neo4j  (graph: who/what/when) ◄──────────────────┤  reads the SAME databases
        ├─► Qdrant (vector: report text)  ◄──────────────────┤
        ▼                                                     ▼
  should_call?  ──yes──► place call / SMS ─────────────► answer text ─► TTS ─► audio to caller
   (criticality, confidence,                                              (Piper or ElevenLabs)
    vitals override)
```

---

## Pipeline 1 — from device signal to a phone call

**Step 1 · Ingest** (`ingest/`). A physiological event arrives as an **HDF5 file** or an **MQTT**
message and is normalized into a `SignalWindow` — raw ECG samples, vitals (HR, BP, SpO₂, RR, temp),
patient history, and window geometry. There is deliberately **no diagnosis yet** (a frozen-contract
rule: the reader never guesses `event_type`).

**Step 2 · The ECG model runs** (`inference/ecg_model.py`). This is the one real neural network:
- **Model:** `ECGTransCovNet` (the "ECG_TransConv" model, vendored from `ecgtranscnn`) — a PyTorch
  CNN + Transformer, loaded from a `.pt` checkpoint.
- **What it does:** `predict(window)` runs the upstream preprocessing → the network → `softmax` →
  `argmax`, yielding **one of 16 arrhythmia classes** (e.g. `ATRIAL_FIBRILLATION`, `NORMAL_SINUS`)
  plus a **confidence** in `[0, 1]`.
- If the checkpoint is missing it falls back to `StubECGModel` so everything downstream still runs.

**Step 3 · Vitals + false-positive gate** (`inference/pipeline.py`). Separately — and **not ML** — the
pipeline computes the **MEWS** score (Modified Early Warning Score: a **rule-based clinical rubric**,
a fixed points table over the vitals) and per-vital **Mann-Kendall trends** (a **classical statistical
test** for a monotonic trend, yielding a direction + p-value — nothing trained or learned). Then the FP gate: a *confident*
`NORMAL_SINUS` (≥ `FP_SUPPRESS_MIN_CONFIDENCE`) is flagged a **false positive**. The output is a
`DeviceEvent` = the window + predicted type + confidence + MEWS + criticality + care guidance + a
markdown report.

**Step 4 · Persist** (`orchestrator/bus_consumer.py`). The event is published to a **Redis stream**
(the bus); a consumer writes it to:
- **Neo4j** (graph): `Patient → HAD_EVENT → MonitoredEvent → HAS_REPORT`, plus conditions, beds, and
  action items — the "who / what / when / relationships" store.
- **Qdrant** (vector): the report narrative is embedded and stored for semantic search. Embeddings
  come from a deterministic **hashing** embedder (default, offline) or **BGE**
  (`bge-small-en-v1.5`) when the `rag` extra is enabled.

**Step 5 · The call decision** (`should_call`, `orchestrator/outbound_flow.py`) — a pure gate:
1. `OUTBOUND_ENABLED` must be on.
2. **Criticality** ≥ `OUTBOUND_MIN_CRITICALITY` (default `High`). Criticality is the more severe of
   the arrhythmia class and the MEWS risk, escalated to at least `High` for any real arrhythmia,
   high MEWS, or a deteriorating trend (`common/criticality.py`).
3. **False-positive gate**: a confident `NORMAL_SINUS` does not call…
4. **Arrhythmia-confidence gate**: …and a *non-normal* prediction only calls when confidence ≥
   `OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE` (default `0.60`) — a low-confidence arrhythmia is likely a
   misdetection.
5. **Vitals override**: a deteriorating patient (MEWS ≥ threshold or a deteriorating trend) calls
   **regardless** of both gates above — vitals beat the rhythm classifier.

If it decides to call, it dispatches a **LiveKit** outbound call (SIP phone or WebRTC room) or an SMS
text, and pushes the event to the **companion app** worklist. The event is **always persisted**; the
gate only governs the *call*.

---

## Pipeline 2 — the voice / chat conversation

Runs in the **LiveKit agent worker** (`voice/livekit_agent.py`). LiveKit carries the audio (WebRTC
for browser, SIP for phone). One turn, in order:

1. **Caller speaks** → audio streams into the LiveKit room.
2. **STT** (speech → text). Swappable via `STT_BACKEND`: self-hosted **faster-whisper** (`base.en`)
   or cloud **ElevenLabs Scribe**.
3. **VAD** (**Silero**) detects end-of-speech so the turn is complete.
4. **Wake-word gate** (follow-ups only): after the first alert, an audio turn must open with
   "hey vios" — unless `AUDIO_WAKE_REQUIRED=false`. The companion app's push-to-talk bypasses this.
5. **PIN gate**: no patient data is spoken until the caller authenticates (fail-closed).
6. **The Handler is "the LLM node" — but it is not an LLM.** LiveKit's `AgentSession` pipeline
   expects an LLM step; a **stub LLM** satisfies that gate, but `llm_node` is overridden to call our
   `Handler` → the **orchestrator** instead. The reasoning is our RAG orchestrator, not a raw model.

**Inside the orchestrator turn** (`orchestrator/orchestrator.py`, `handle_turn`):
1. **Input guardrail** — refuse unsafe / out-of-scope requests *before* any retrieval or model call.
2. **Retrieve (RAG)** via `HybridRetriever`: a **vector** search over Qdrant (*Retrieved passages*)
   **plus** a **graph** query over Neo4j (*Known relationships*). Two separately-labelled,
   separately-cited blocks (`RetrievalResult`) — no cross-block re-ranking.
3. **De-identify** the retrieved context (Presidio or regex, `DEID_BACKEND`) so no name/PHI reaches
   the model.
4. **LLM generate**: the prompt goes to the `LLMProvider` — self-hosted **Ollama** (`llama3.2`) by
   default, wrapped in `DeidentifyingLLM`. `EchoLLM` is used offline in tests; cloud
   Anthropic/OpenAI are swappable but only on synthetic data.
5. **Output guardrail** → the final answer text.

**Back to voice:**
7. Answer text → **TTS** → audio to the caller. Swappable via `TTS_BACKEND`: **Piper** (self-hosted)
   or **ElevenLabs**. Before *cloud* TTS the text is de-identified again (defense in depth), and
   machine tokens are normalized (`speakable`, `voice/adapters.py`) so `NORMAL_SINUS` is spoken as
   "normal sinus", not letter-by-letter with the underscore.
8. Saying **"acknowledge"** flips the event's status in Neo4j — closing the outbound loop.

Two shortcuts reuse the same machinery:
- **Speak-on-select** (companion app): selecting a worklist row sends a `/select` control message;
  the worker speaks that event's stored `Report.summary` via TTS (`INBOX_SPEAK_ON_SELECT`).
- **Text chat**: a typed question runs the same orchestrator turn and returns a **typed** answer (no
  STT/TTS). Typed turns are never wake-word gated.

### The voice worker — what joins the room

`cli.voice_worker` → `voice/livekit_agent.py` is a long-running **agent-worker process**, separate
from the LiveKit server. The split of responsibility is the key idea:
- **LiveKit server** (`:7880`) = **transport only** — it moves audio/data packets between
  participants (WebRTC for browser, SIP for phone). It knows nothing about ECG, PINs, or the KB.
- **The worker** = the **intelligence** — it joins a room *as a participant* and runs the whole
  `STT → gates → Handler/orchestrator → TTS` loop for that call. It is what listens, thinks
  (grounded in the KB), and speaks.

**Lifecycle:** start → connect to LiveKit + register as agent `rmsai-agent` → **wait** → a dispatch
arrives for a room → **join** it (`ctx.connect`) → run the `AgentSession` job for the whole call →
cleanup → back to wait. One process serves **many rooms over its lifetime**; because it registers
under a name it uses **explicit dispatch** — it only joins rooms it is told to (the gateway on
`/session`, the consumer per outbound event, or `cli.dispatch`), never auto-joining.

**Process vs. job — what is "always on" and what is per-event.** Distinguish the two:
- The **worker process** (`cli.voice_worker`) is the single, **always-on** host. It registers once and
  waits; it is *not* per-event and only dies if you stop/restart it.
- A **job** is one room's session. When a room is dispatched, the framework runs that room's
  `_entrypoint` in its **own prewarmed subprocess** (`_prewarm` warms the VAD + Ollama there); the job
  joins the room, runs the loop, and exits when the room ends. This subprocess join is what pays the
  ~40s cold-start.

A job's lifespan therefore equals its **room's** lifespan:

| Room | Job (the per-room "instance") |
|---|---|
| **per-event outbound** (`rmsai-outbound-<event_id>`) — ephemeral, closes when the call ends | **created on dispatch, torn down when the call ends** |
| **companion-app inbox** (`rmsai-inbox-<hospital>`) — persistent while the app is connected | **long-lived** — stays as long as the inbox room/app session exists |

So there is **no separate "shared worker" vs "per-event worker" as distinct processes** — it is one
always-on process running many jobs (one per room), each in its own subprocess: an ephemeral room ⇒ a
short job, a persistent room ⇒ a long job. Restarting the worker process kills **all** its jobs
(including the long-lived inbox job), which is exactly why the persistent inbox room needs a fresh
dispatch after a restart — the auto-redispatch / `cli.dispatch` that closes that gap.

On joining, `_entrypoint` reads the room name and picks the Handler:
- `rmsai-inbox-*` → `InboxHandler` (companion app: push-to-talk, selection-scoped Q&A, speak-on-select);
- an outbound event room with a staged `OutboundAlert` in **Redis** → `OutboundHandler` (speaks that
  event's alert after the PIN, then Q&A + acknowledge);
- otherwise → inbound KB Q&A.

That third bullet is the seam where the **event pipeline hands off to the conversation pipeline**: the
consumer placed the call and wrote *why* into Redis; the worker — a **separate process** — reads that
alert on join and voices the right event.

**Two-way audio, one room.** Within **whichever single room it joins**, the worker is a **full
bidirectional participant** — it does *not* listen in one room and speak in another. In that one room
it both:
- **subscribes** to the human's audio track (their speech) → feeds **STT**, and
- **publishes** its own audio track (**TTS**) → the human hears the answer.

This dual behavior is identical in both room types, they are just independent conversations:
- **inbox room** (`rmsai-inbox-<hospital>`) — subscribes to the app user's push-to-talk audio **and**
  publishes TTS, both here;
- **per-event room** (`rmsai-outbound-<event_id>`) — a *separate* worker job subscribes to the remote
  clinician's speech (phone via SIP or browser via WebRTC) **and** publishes TTS, both here.

So "the worker delivers TTS to the event's room and subscribes to it for the caller's speech" is
correct — and the *same* two-way exchange happens in the inbox room for the app. (The worker can serve
both at once, as two independent jobs; each is a self-contained two-way session in its own room.)

### Room models: inbox vs outbound, and SIP vs WebRTC

There are two shapes of room. The worker and its loop are **identical** in both — only *how the human
gets into the room* differs.

| | Companion-app **inbox** | **Outbound event call** |
|---|---|---|
| Room name | `rmsai-inbox-<hospital_id>` — **one persistent room per hospital** | `rmsai-outbound-<event_id>` — **one room per event** |
| Lifespan | long-lived; many events flow through it | ephemeral; created for the alert, closes when the call ends |
| Which event | selection scopes it (`/select`) | the room *is* the event; its alert is staged in Redis |
| Transport | WebRTC (browser) | **SIP (phone)** or **WebRTC (browser)** |

The **two outbound transports** both use the same per-event room (`rmsai-outbound-<event_id>`) and then
run the same worker loop — they differ only in **who initiates the audio connection**:

- **SIP (phone) — relay-initiated.** The relay actively **dials the clinician's phone number** through
  a SIP trunk (`LiveKitCaller.place_call` → `create_outbound_sip_call`); LiveKit bridges the PSTN call
  into the room. The clinician's phone rings. Needs `LIVEKIT_SIP_TRUNK_ID` + a number +
  `OUTBOUND_ENABLED`.
- **WebRTC (browser) — clinician-initiated.** The relay only **stages the alert and prints a join
  link/token** (or pushes it to the app); the clinician clicks and joins from a browser. No dialing,
  no trunk, no phone (`caller_factory` is a no-op `SimulatedCaller`).

So your intuition is right: **each outbound event gets its own room** — that per-event room carries
that event's staged alert, whether the human arrives by SIP or WebRTC. The companion-app **inbox is the
exception**: one shared, persistent room where selection picks the event. (The static `rmsai-outbound`
room, `LIVEKIT_SIP_ROOM`, is only the *default* when a `LiveKitCaller` is built without an explicit
room — the standalone `cli.outbound` path, one call at a time; the bus consumer always uses a per-event
room for both transports.)

**The companion app is entirely single-room** — it uses only `rmsai-inbox-<hospital_id>` for
everything, and never joins a per-event room:
- **Worklist (listing)** is **push-only**: the consumer publishes each critical event as a data
  message *directly via the LiveKit server API* (`live/inbox.py`, `RoomService.send_data`) — pushed by
  the **relay, not the voice worker**, so the list populates even with no worker present. The app's
  `DataReceived` handler feeds a pure reducer (`applyMessage`) that builds the table; acknowledgements
  arrive the same way as `type:"status"` messages.
- **Selecting a row does NOT join that event's room.** It sends a `/select <event_id>` control message
  on the chat topic in the *same* inbox room, which scopes the worker's conversation to that event.
  Chat answers (text stream), the spoken summary / voice answers (the worker's audio track), and inline
  artifacts (`type:"show"` + scoped HTTP token links) all flow inside that one room.

### Notification vs. outbound call — why per-event rooms exist

A natural question: if events are already posted to the app's inbox room, why do per-event rooms exist
at all? Because **posting a notification and placing a call are two different jobs**:

| | Inbox data message | Per-event outbound room |
|---|---|---|
| What it is | a **notification** (metadata) | a **live audio call session** |
| Goal | "show this in the worklist" | "**reach** a clinician who isn't watching the app" |
| Mechanism | a `send_data` packet into the shared room | a room the clinician is **dialed into** (SIP) or **joins by link** (WebRTC) |
| Participants | the app (already connected) | the agent worker **+** that one clinician |

For a clinician **using the app**, events already *are* delivered to their room (the worklist), and
voice Q&A happens **in that same room** — no separate room needed. The per-event rooms exist for the
*other* case: **actively reaching a clinician who is not in the app** — ringing their **phone** (SIP)
or sending a **join link** (WebRTC) for a critical event. You cannot "post a data message" to make a
phone ring; a call is a media session that must be *placed*, and it needs a room to live in.

**Who subscribes to a per-event room:** exactly two participants — a private 1:1 call.
1. the **voice worker** (dispatched in; reads the event's `OutboundAlert` from Redis to know what to
   voice), and
2. the **clinician** (their phone bridged in by the SIP trunk, or their browser joined via token).

**Why it can't just reuse the shared inbox room:**
- **Isolation / PHI** — the inbox room is a per-hospital *shared* space; bridging live calls about
  different patients into it would let everyone hear everyone (cross-talk, PHI leakage). Each 1:1 call
  must be isolated.
- **Routing** — the room name *is* the event id, which is how the worker (and the Redis alert) know
  *which* patient/event the call is about. A shared room can't carry that.
- **Lifecycle** — a call is short-lived (ring → answer → alert → Q&A → hang up); a dedicated room gives
  it clean create/teardown, independent of the long-lived inbox room.

**Mental model:** *push a notification to someone already looking* (inbox room) vs. *place a call to
reach someone who isn't* (per-event room). The second fundamentally needs its own isolated media room
per event.

---

## The models, at a glance

| Role | Model / tech | Where | Swappable via |
|---|---|---|---|
| **ECG classification** | `ECGTransCovNet` (CNN + Transformer, PyTorch) | `inference/ecg_model.py` | checkpoint; stub fallback |
| **Vitals / severity** | MEWS (rule-based rubric) + Mann-Kendall trends (statistical test) — **no ML** | `inference/`, `common/criticality.py` | — |
| **Embeddings** (vector search) | hashing (default) or **BGE** `bge-small-en-v1.5` | `kb/vector/` | `EMBEDDER` |
| **LLM** (answers) | **Ollama `llama3.2`** (local); Echo offline | `common/providers.py` | `LLM_PROVIDER` (`LLMProvider` interface) |
| **STT** (speech→text) | faster-whisper `base.en` or ElevenLabs Scribe | `voice/adapters.py` | `STT_BACKEND` |
| **TTS** (text→speech) | Piper or ElevenLabs | `voice/adapters.py` | `TTS_BACKEND` |
| **VAD** (speech detection: endpointing + barge-in) | Silero | LiveKit worker | — |
| **De-identification** | Presidio (NER) or regex | `common/deid.py` | `DEID_BACKEND` |
| **Graph DB** | Neo4j | `kb/graph/` | — |
| **Vector DB** | Qdrant | `kb/vector/` | — |
| **Voice transport** | LiveKit (WebRTC + SIP) | `voice/` | — |

---

## Design invariants

- **Self-hosted for PHI, cloud only for synthetic.** Everything that touches real patient data runs
  locally (whisper, piper, ollama, neo4j, qdrant). Cloud models (ElevenLabs, Anthropic/OpenAI) are
  permitted only on synthetic data (hard rules #4/#5).
- **One interface per model.** `ECGModel`, `LLMProvider`, `STTAdapter`/`TTSAdapter`, `EventStore`,
  etc. Any single model swaps by config without touching the flow.
- **Redaction by construction.** Patients are referenced by pseudonym everywhere (including logs);
  de-identification runs before every model call and before any cloud TTS, and **fails closed**.
- **Frozen contracts.** `SignalWindow` (no predicted type), `DeviceEvent`, `RetrievalResult` (two
  cited blocks) are defined in `common/` and decouple the stages from each other.
- **Resilience.** Graph traversals and RAG chains tolerate missing nodes / broken chains without
  throwing — a partial KB degrades the answer, it doesn't crash the call.
