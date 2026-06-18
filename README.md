# rmsai-relay

Self-hostable POC for **medical IoT + AI**. It ingests physiological event data (HDF5 archives / an
MQTT stream), classifies clinically-significant arrhythmias with the `ECG_TransConv` model, persists
each event into a graph + vector knowledge base, **calls a remote clinician over the phone** about
critical events, and then answers the clinician's follow-up questions — by voice or text — grounded
in a clinical knowledge base and a per-patient knowledge graph.

Built **leaf-up, test-first, one phase at a time**. Every subsystem ships a `pytest` suite and a
runnable CLI harness before it is wired upward. See [`CLAUDE.md`](CLAUDE.md) for the full working
agreement; the source-of-truth design lives in the project spec + kickoff prompt.

---

## Background

Bedside monitors emit a continuous stream of physiological signals (ECG leads, SpO₂, respiration)
plus vital signs. Most of that is noise; a small fraction are clinically actionable arrhythmias
(atrial fibrillation with RVR, VT/VF, etc.). This POC closes the loop from **signal → detection →
clinician** and keeps a conversational, evidence-grounded channel open afterward:

1. **Detect** the significant event (the vendored `ECG_TransConv` classifier over all 7 ECG leads).
2. **Contextualize** it — false-positive gating, MEWS scoring, Mann-Kendall vital-trend analysis,
   care guidance, and a markdown clinician report.
3. **Persist** it into a knowledge base — a Neo4j graph (patient ↔ event ↔ condition ↔ guideline)
   and a Qdrant vector store (clinical-protocol passages + report narrative).
4. **Report** critical events by placing an outbound voice call (LiveKit/SIP) or a text message.
5. **Converse** — after a shared-PIN gate, the clinician asks follow-up questions answered from the
   KB + that patient's graph, and can verbally **acknowledge** the event (flips its status).

### What "done" means (POC success criteria)

1. A raw HDF5 archive can be analysed from the CLI and produces a `DeviceEvent` with a
   model-predicted `event_type`.
2. A clinician can query the preloaded KB in **text** and get correct, **cited** answers grounded in
   both document content (vector) and entity relationships (graph).
3. The same works over a **phone call**, after caller authentication, within a usable latency budget.
4. A detected non-false-positive event triggers an **outbound** call to a single preconfigured
   number, delivers a spoken report, and supports an interactive grounded follow-up that is
   acknowledged and recorded.
5. **PHI never leaves the local boundary** and is provably de-identified before any model call.

### System model — three planes over one orchestrator

- **Telemetry plane** — HDF5/MQTT **reader** → `SignalWindow` → `ECG_TransConv` (`ECGModel`) →
  `DeviceEvent` → event bus.
- **Knowledge plane** — a **hybrid** retriever that per query runs **vector search** over document
  chunks **and** a **graph lookup** of entity relationships, fusing both into one labelled context
  for the LLM (no cross-block re-rank, no standalone graph mode, no Wiki path).
- **Interaction plane** — a voice surface (SIP/LiveKit → STT/TTS behind an auth gate), a text-chat
  surface, and (planned) a companion app for visual/streaming content. Voice & chat are the
  **control plane**; the companion app is the **data plane** for waveforms/video voice can't carry.

### Design principles (hard rules)

- **Self-hosted by default; provider abstraction always.** All real/PHI processing runs against
  local models. The LLM sits behind one `LLMProvider` interface (`OllamaProvider` default;
  `Anthropic`/`OpenAI` swappable). Cloud APIs are used **only** on synthetic data, never PHI.
- **Synthetic data only in development.** PHI never reaches a third-party API and is never written
  to plain-text logs.
- **Redaction by construction.** Patients are referenced by `id`/`pseudonym` everywhere, including
  logs. Names and free-text notes are never logged.
- **Fail closed on safety, degrade gracefully elsewhere, never crash the pipeline, always log.**
  De-id failure aborts the turn; missing graph nodes / broken RAG chains yield caveated answers, not
  exceptions; out-of-corpus questions decline rather than fabricate.

---

## High-level data-flow architecture

```
                          ┌──────────────────────────────────────────────────────────────┐
                          │                      INGEST / DETECT                           │
  HDF5 archive  ──┐       │  ingest/ (HDF5 reader, MQTT)  →  SignalWindow                  │
  MQTT stream   ──┴──────►│  inference/ ECG_TransConv  →  FP gate → MEWS → vital trends    │
                          │                              →  care guidance → markdown report │
                          │                              =  DeviceEvent                     │
                          └───────────────┬──────────────────────────────────────────────┘
                                          │  event_to_dict()
                       cli.ingest --emit  │  (XADD)
                                          ▼
                                ┌───────────────────┐
                                │  Redis Stream      │   bus =  rmsai.events
                                │  "rmsai.events"    │   (consumer group, exactly-once ack,
                                └─────────┬─────────┘    partition by patient_id → ordering)
                       cli.consume        │  XREADGROUP
                       (dict_to_event)    ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                            ORCHESTRATE  (orchestrator/)                         │
        │  ensure_patient (G8 auto-create: bed assign + synthetic history → graph)        │
        │  process_device_event   ── persist ──►  Neo4j MonitoredEvent + ActionItems      │
        │                          ── archive ──►  Qdrant report narrative + Report node   │
        │  criticality(event_type, mews_risk) → should_call gate                          │
        │        │ below threshold / NORMAL_SINUS → skip (still persisted)                │
        │        ▼ critical                                                               │
        │  dispatch:  voice (LiveKit/SIP or WebRTC)   |   text (SMS notifier)             │
        └───────────────────────────────┬────────────────────────────────────────────────┘
                                         │  stage OutboundAlert (Redis, keyed by room)
                                         ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                         VOICE LOOP  (two-process model)                         │
        │  relay PLACES the call (LiveKitCaller → SIP dial / WebRTC room)                  │
        │  cli.voice_worker JOINS the room:                                               │
        │     audio ──Whisper STT──► OutboundHandler ──Ollama LLM (RAG)──► Piper TTS ──►   │
        │     PIN gate → speak THIS event's alert → Q&A grounded in KB+graph → "ack"       │
        │     → MonitoredEvent.status = acknowledged                                       │
        └──────────────────────────────────────────────────────────────────────────────┘

  Inbound path (clinician calls/joins to query): same worker, build_handler() → PIN gate →
  intent → operational Cypher template OR hybrid retrieve → de-id → LLM → cited answer.
```

### Frozen contracts (`common/schemas.py`, `common/interfaces.py`)

- **`SignalWindow`** — reader output (multi-rate signals in mV, vitals + history, window geometry,
  per-signal quality). **No predicted `event_type`**; sim files may carry a ground-truth `condition`
  attr for eval only.
- **`DeviceEvent`** — `SignalWindow` + model-predicted `event_type`/`confidence`,
  `is_false_positive` (⇔ `NORMAL_SINUS`), `ClinicalAnalysis` (MEWS + per-vital trend + correlations),
  `care_guidance`, and the markdown `report_md`.
- **`RetrievalResult`** — two labelled blocks: *Retrieved passages* (vector) + *Known relationships*
  (graph), separately cited; relationships empty under `vector` mode.
- **Interfaces**: `LLMProvider`, `ECGModel`, `VitalsAnalysis`, `EventStore` (+ `BedAssignment` /
  `PatientHistory` stubs). `event_type` ∈ the 16 `ecgtranscnn` classes; `NORMAL_SINUS` ⇒ FP.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language / runtime | Python ≥3.10, managed with **`uv`**; `pytest` + `ruff`, type hints throughout |
| Contracts | **pydantic v2** schemas (`common/schemas.py`) |
| ECG classifier | **`ECG_TransConv`** (vendored `crtx-sg/ecgtranscnn`, PyTorch CPU), wrapped never reimplemented |
| Vitals analysis | `ecgtranscnn` MEWS + Mann-Kendall trend + ECG-vital correlation (statistical) |
| Event bus | **Redis Streams** (`rmsai.events`, consumer groups; partition by `patient_id`) |
| Graph KB | **Neo4j** + Cypher (patient ↔ event ↔ condition ↔ treatment ↔ guideline ↔ bed/unit) |
| Vector KB | **Qdrant** + embeddings (BGE via `sentence-transformers`, deterministic Hashing fallback) |
| Memory tiers | working (Redis) · episodic (Qdrant) · semantic (= vector KB) |
| LLM | **Ollama** (self-hosted, default) behind `LLMProvider`; Anthropic/OpenAI swappable on synthetic data |
| De-identification | Regex (default) or **Presidio** + spaCy (`deid` extra), fail-closed before any model call |
| Speech (self-hosted) | **faster-whisper** STT + **Piper** TTS + **silero** VAD |
| Telephony / WebRTC | **LiveKit** (agent worker + SIP outbound + browser WebRTC) |
| EMR | **HAPI FHIR** (`emr/`, stub → real in Phase 8) |
| Orchestration | LangGraph-style turn orchestrator (`orchestrator/`) |
| Infra | Docker Compose (`infra/docker-compose.yml`): neo4j, qdrant, redis, mosquitto, model-server (ollama), hapi-fhir, livekit |

**No SQL DB in the POC.** Five stores: Neo4j (relationships + operational event log, behind an
`EventStore` repository interface so it can migrate to Postgres/TimescaleDB later), Qdrant (text +
embeddings), Redis (working memory + bus), HDF5 (waveforms), object/file store (plots, reports).

### Repository layout

`common/` contracts + config + de-id + protocols · `ingest/` HDF5 + MQTT readers · `inference/`
model + vitals + serialize · `kb/{vector,graph,hybrid}` retrieval · `memory/` working/episodic tiers
· `orchestrator/` turn loop + outbound flow + bus consumer · `voice/` SIP/LiveKit + handlers +
STT/TTS · `emr/` FHIR · `app/`+`live/` companion app & live media (Phase 9, planned) · `cli/`
entrypoints · `infra/` compose · `external/ecgtranscnn/` vendored model+simulator (gitignored) ·
`data/{synthetic,fixtures}` · `docs/` clinical corpus.

---

## Phase status

Built leaf-up, one phase at a time; a phase is "done" only when its CLI test passes. Phases **0–8
are complete** (see `git log`); **Phase 9 is planned/deferred** until the core loop is hardened.

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Foundations — infra, frozen contracts, synthetic generator, vendored `ecgtranscnn` | ✅ |
| 1 | Reader → `SignalWindow`; `ECGModel`; FP gate; `VitalsAnalysis` → `DeviceEvent`; HDF5/MQTT CLI | ✅ |
| 2A | Vector RAG baseline (chunk → BGE → Qdrant) | ✅ |
| 2B | Graph: patient records + document-entity extraction + operational Cypher templates | ✅ |
| 2C | Hybrid retriever (labelled side-by-side passages + relationships) | ✅ |
| 2D | Evaluation harness (`hybrid` vs `vector`) | ✅ |
| 3 | Memory tiers (working / episodic / semantic) | ✅ |
| 4 | Orchestrator over text + event persistence, report assembly & archival | ✅ |
| 5 | Voice infra as echo bot (LiveKit + Whisper + Piper) | ✅ |
| 6 | Voice + orchestrator inbound (PIN-gated spoken grounded answers) | ✅ |
| 7 | Outbound full loop (event → call → grounded follow-up → ack) | ✅ |
| 8 | Hardening — tracing, guardrails, real HAPI FHIR, failure-mode tests | ✅ |
| 9 | Companion app + live media (MQTT→WebRTC ECG/vitals, camera relay, consent/audit) | 🔜 planned |
| — | Bus consumer + event-driven LiveKit/WebRTC outbound (`cli.consume`) | ✅ |
| — | File-drop auto-publish watcher (inotify → `cli.ingest --emit bus`) | 🔜 planned |

---

## Knowledge-base data model

The graph is the workhorse for operational queries; the vector store holds protocol documents and
archived report narrative. Key nodes/edges:

- **Nodes:** `Patient`, `Condition`, `Treatment`, `Symptom`, `Surgery`, `Guideline`, `Unit`, `Bed`,
  `MonitoredEvent` (persisted `DeviceEvent` with inline vitals snapshot + criticality + lifecycle
  status + artifact refs), `ActionItem`, `Report`, `CareProtocol`/`ProtocolStep`.
- **Edges:** `HAS_DIAGNOSIS`, `CO_MORBID_WITH` (derived from cohort co-occurrence, carries
  confidence/count/window), `PRESENTS`, `HAD_SURGERY`, `PRESCRIBED`, `MANAGES`, `ASSIGNED_TO`
  (Patient→Bed), `IN_UNIT`, `HAD_EVENT`, `AT_BED`, `OF_CONDITION`, `FOLLOWED_BY` (per-patient chain
  ordered by `event_timestamp`), `HAS_ACTION`, `HAS_REPORT`, `APPLIES_TO`, `HAS_STEP`.

**Care protocols** are curated external YAML (`common/protocols/care_protocols.yaml`), matched on
`event_type` + vital conditions + min severity (most-specific-wins, with a default fallback), loaded
into the graph **and** indexed as narrative into the vector store.

### Operational query matrix (the verification target)

Every row is answerable through tested, read-only, parameterized Cypher templates (free
text-to-Cypher is an allowlisted read-only fallback only):

| # | Query | Store |
|---|-------|-------|
| 1 | Critical events last 24h by patient/bed/unit | graph |
| 2 | Positive (non-FP) events last *x* min | graph |
| 3 | Event status on a bed (ts, event, FP?, actual condition) | graph |
| 4 | Event analysis report for a patient/bed/unit | graph + vector |
| 5 | Vitals at the time of a specific event | graph (inline snapshot) |
| 6 | Outstanding action items across patients | graph |
| 7 | Care protocol for a bed's last event | hybrid |
| 8 | Patterns: age/gender/co-morbidity/symptom → event type | graph (analytics) |
| 9 | ECG strips for last AFib event | graph → artifact ref |
| 10 | HR & BP trend for last Tachycardia event | graph → artifact ref |

### Example questions the system answers

These are the natural-language prompts a clinician can ask (over text or an authenticated call) that
the data model is built to serve. Each maps to a tested query template (`T*` above) or the hybrid
retriever:

| Question | Maps to |
|----------|---------|
| "List all patients, bed number, unit/ward who had **critical events in the last 24 hours**." | T1 (graph) |
| "List **positive patient events** reported in the **last *x* minutes**." | T2 (graph, non-FP) |
| "What is the **status of events reported on Bed xx** — timestamp, reported event, false positive?, actual condition?" | T3 (graph) |
| "Get the **event analysis report** for patient / Bed xx in Unit/Ward." | T4 (graph → vector report content) |
| "What were the **vitals at the time of the specific event** for the patient in Bed xx?" | T5 (graph, inline vitals snapshot) |
| "Provide an **action-item list** of all outstanding actions for patients." | T6 (graph) |
| "What is the **treatment / care protocol** for Bedside x's last reported event?" | T7 (hybrid: graph condition + protocol, narrative from vector) |
| "From the data, do you see any **pattern of age / gender / co-morbidities / symptoms** leading to a specific event?" | T8 (graph analytics — framed as correlation, not causation) |
| "Show me the **ECG strips** for the patient with the last reported AFib event." | T9 (graph → artifact ref; companion app / chat link, spoken summary over voice) |
| "Show me the **HR and BP trend** for the patient with the last reported Tachycardia event." | T10 (graph → vitals/trend plot ref) |
| "Show **all patients who have a specific event** (e.g. all AFib)." | graph traversal: `MonitoredEvent {event_type}` → `Patient` (+ optional bed/unit) |

> Operational items (1–6, 9, 10) are judged by **exact-row match**; hybrid/relationship items (7, 8)
> by **groundedness + citations**. Out-of-corpus / unknown-bed / unknown-patient questions are
> expected to **decline**, never fabricate.

---

## POC configuration & stubs

Everything synthetic/simplified sits behind an interface so the real implementation swaps in without
touching callers.

**Key config defaults** (all env/`.env`-overridable):

| Key | Default | Purpose |
|-----|---------|---------|
| `FP_SUPPRESS_MIN_CONFIDENCE` | `0.80` | suppress as FP only if `NORMAL_SINUS` ≥ this; else flag `uncertain` |
| `LOW_CONFIDENCE_CAVEAT` | `0.60` | top-class confidence below this marks the event `low_confidence` |
| `CRITICALITY_NORMAL_EVENT` | `NORMAL_SINUS` | the only event treated as non-critical; any other event ⇒ at least High |
| `CRITICALITY_MEWS_THRESHOLD` | `3` | MEWS score at/above this ⇒ escalate criticality to High |
| `CRITICALITY_ESCALATE_ON_DETERIORATING` | `true` | any deteriorating vital trend ⇒ escalate criticality to High |
| `OUTBOUND_ENABLED` / `OUTBOUND_MIN_CRITICALITY` | `false` / `High` | gate which events dial out |
| `OUTBOUND_CALL_NUMBER` / `OUTBOUND_FROM` | — | single hard-configured destination + caller ID |
| `OUTBOUND_MAX_RETRIES` / `OUTBOUND_RETRY_DELAY_S` | `2` / `30` | no-answer retry policy |
| `INBOUND_AUTH_PIN` | shared PIN | verified before any PHI is voiced |
| `DEID_BACKEND` | `auto` | `auto` / `regex` / `presidio` |

**Stubs (POC → production):** `PatientHistory` (synthetic, seeded by `patient_id` → EMR/FHIR) ·
`BedAssignment` (≤25 beds/unit, overflow, clear ops → ADT feed) · `ECGModel` (vendored checkpoint or
deterministic test stub → validated SaMD model on Triton) · `EventStore` (Neo4j → Postgres/
TimescaleDB at scale) · FHIR client (stub → HAPI) · outbound (single number → escalation tree) ·
caller auth (shared PIN → per-user identity/MFA) · audit (JSONL → tamper-evident store).

### Safety & PHI guarantees

- No PHI in logs (redacting logger; patients referenced by pseudonym only).
- De-identification asserted on model inputs in tests; **fails closed** (no LLM call on de-id error).
- Caller authentication enforced before any PHI is voiced; LLM-generated Cypher constrained to
  read-only.
- Append-only **JSONL audit log** from Phase 0: `{ts, actor, action, subject(pseudonym), outcome}`
  for PHI reads, outbound calls + outcome, inbound auth results, and queries.
- Idempotent ingestion (MERGE by event `uuid`) dedupes MQTT+HDF5 and replays.

---

## Setup

```bash
# 1. Vendor the ECG model + simulator (gitignored; see external/ecgtranscnn/PLACEHOLDER.md)
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn

# 2. Install (uv). The vendored package is installed editable separately — `uv sync` does not
#    track it (it is gitignored), so re-run the second line after any `uv sync`.
uv sync --extra dev
uv pip install -e external/ecgtranscnn

# 3. Bring up the lean datastores
docker compose -f infra/docker-compose.yml up -d redis neo4j qdrant

# 4. Run tests
uv run pytest
```

### Optional extras

```bash
uv sync --extra rag                                   # real BGE embeddings + reranker
uv sync --extra deid && uv run python -m spacy download en_core_web_sm   # Presidio de-id
uv sync --extra voice                                 # faster-whisper STT + Piper TTS
uv sync --extra livekit                               # LiveKit agent worker + SIP/WebRTC
```

`.env` keys that matter: `REDIS_URL`, `NEO4J_*`, `QDRANT_URL`, `DEID_BACKEND`, `STT_BACKEND`/
`TTS_BACKEND`, `INBOUND_AUTH_PIN`, and for live calls `LIVEKIT_URL` / `LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET` plus `OUTBOUND_ENABLED=true`.

> If you hit `ModuleNotFoundError: presidio_analyzer`, set `DEID_BACKEND=auto` (or `regex`), or
> install the `deid` extra above.

---

## End-to-end testing

### 1. Tests (offline, no infra)

```bash
uv run pytest -q                       # full suite
uv run pytest -q -m "not infra"        # skip infra-dependent tests
```

### 2. Generate synthetic events (vendored simulator)

```bash
# writes HDF5 under external/ecgtranscnn/data/inference/
uv run python external/ecgtranscnn/scripts/generate_inference_data.py
```

### 3. Direct relay path (HDF5 → call, bypasses the bus)

```bash
uv run python -m cli.outbound \
    --file external/ecgtranscnn/data/inference/<file>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt
```

### 4. Bus path: producer → Redis Stream → consumer

```bash
# PRODUCER — classify + publish to rmsai.events
uv run python -m cli.ingest \
    --file external/ecgtranscnn/data/inference/<file>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt \
    --emit bus --stream rmsai.events

# CONSUMER — persist (graph+vector) + criticality-gated dispatch; drain backlog then exit
uv run python -m cli.consume --channel voice --once \
    --follow-up "what were the vitals at the event" --ack "yes I acknowledge"

# text channel instead of voice
uv run python -m cli.consume --channel text --once
```

> Run a **single** consumer per patient — concurrent consumers on the same patient cause Neo4j
> write-lock timeouts (poison-handled, but those events get skipped). Partitioning the bus by
> `patient_id` keeps ordering while patients run in parallel.

### 5. Real WebRTC audio loop (browser, no phone)

```bash
# TERMINAL A — agent worker (joins rooms; runs Whisper STT → Handler → Piper TTS)
uv run python -m cli.voice_worker dev
```

**Inbound (ask the KB):**
```bash
uv run python -m cli.livekit_token --room rmsai-call-demo
# → open https://agents-playground.livekit.io → Manual → paste URL + Token → allow mic
# → speak the PIN → ask a KB question
```

**Outbound (event-driven, carries the event context):**
```bash
# produce an event (step 4 producer), then stage the alert + print a join token
uv run python -m cli.consume --channel voice --caller livekit --transport webrtc --once
# → join the printed rmsai-outbound-<event_id> room in the playground
# → PIN → hear THIS event's alert → Q&A → say "acknowledge" (flips event status)
```

### 6. Real SIP phone call (outbound to a number)

```bash
# requires SIP trunk/number in .env + OUTBOUND_ENABLED=true; worker (step 5A) running
uv run python -m cli.consume --channel voice --caller livekit --transport sip --once \
    --number +1XXXXXXXXXX
```

### Recommended smoke test (clean, single pass)

```bash
uv run python external/ecgtranscnn/scripts/generate_inference_data.py
uv run python -m cli.ingest --file external/ecgtranscnn/data/inference/<file>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt --emit bus
uv run python -m cli.consume --channel voice --once \
    --follow-up "what were the vitals" --ack "yes I acknowledge"
```

Expect: critical events (e.g. AFib/High) persisted + called + acknowledged; NormalSinus/Low
persisted but skipped (`below_threshold`).

### Other CLI harnesses (per-subsystem)

```bash
uv run python -m cli.gen_synthetic ...     # synthetic signal/event generation
uv run python -m cli.kb_vector index --dir docs/   # index the clinical corpus (vector)
uv run python -m cli.graph migrate         # graph schema migrate / seed
uv run python -m cli.kb ask "..."          # hybrid (vector + graph) KB query
uv run python -m cli.kb_eval               # vector vs hybrid over the gold question set
uv run python -m cli.memory demo           # working + episodic + semantic memory round-trip
uv run python -m cli.speech_check          # offline Piper TTS → Whisper STT round-trip
uv run python -m cli.text_chat             # inbound query by text (PIN-gated, KB-grounded)
uv run python -m cli.voice                 # offline typed-text voice demo (no audio hardware)
```
