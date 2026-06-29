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
        │  cli.voice_worker JOINS the room (Handler is the 'LLM' node; stub LLM fills the  │
        │  pipeline gate so llm_node runs):                                                │
        │   audio ─Whisper STT─►[wake word]─► OutboundHandler ─Ollama LLM(RAG)─► Piper TTS │
        │   text  ─chat box────────────────► OutboundHandler ─Ollama LLM(RAG)─► chat text  │
        │     PIN gate → speak THIS event's alert → Q&A grounded in KB+graph → "ack"       │
        │     → MonitoredEvent.status = acknowledged                                       │
        │  modality-matched: audio→audio (wake-gated), text→text; never crossed            │
        └──────────────────────────────────────────────────────────────────────────────┘

  Inbound path (clinician calls/joins to query): same worker, build_handler() → PIN gate →
  intent → operational Cypher template OR hybrid retrieve → de-id → LLM → cited answer.
```

### Voice worker architecture (the VOICE LOOP, explained)

The worker (`cli.voice_worker` → `voice/livekit_agent.py`) joins a LiveKit room and runs the call.
A few design points that aren't obvious from the diagram:

- **The `Handler` *is* the "LLM node", not a model.** LiveKit's `AgentSession` pipeline is
  STT → *LLM* → TTS. We override that LLM node (`HandlerAgent.llm_node`) to call our conversation
  `Handler` instead. The Handler runs the **PIN gate → de-identification → KB/graph retrieval →
  grounded answer** path — i.e. all the safety and grounding logic lives here, *not* in a raw model
  prompt. The real LLM (**Ollama**, local) is still used, but **inside** the Handler/orchestrator
  (RAG over the clinical KB + per-patient graph), one layer below the pipeline.
- **The "stub LLM" is a permanent shim, not a placeholder.** LiveKit *skips reply generation
  entirely* when `session.llm is None`, so `llm_node` would never run. We install a no-op
  `make_stub_llm()` purely to satisfy that gate; its `chat()` is never called. This is **not** a
  temporary fix awaiting a "real LLM" — Ollama is already the real LLM (via the Handler). Swapping
  providers means changing `LLM_PROVIDER` (the orchestrator's provider), never this shim.
- **Modality-matched I/O.** *Audio* turns go STT → wake-word gate → Handler → **TTS (audio)**.
  *Text* turns (LiveKit chat box) go through a separate `text_input_cb` → Handler → **`send_text`
  (text on the chat channel)**, bypassing TTS. Audio→audio, text→text; they never cross.
- **Wake word gates *audio only*.** After the alert, follow-up **audio** Q&A must start with
  `AUDIO_WAKE_WORD` ("hey vios"); the agent then stays awake for `AUDIO_WAKE_WINDOW_S` (30 s) so
  back-and-forth speech needn't repeat it. This guards against room noise and Whisper
  hallucinations-on-silence. **Text chat is never wake-word gated** — `AUDIO_WAKE_WINDOW_S` does not
  apply to typing. The PIN, the spoken alert, and the verbal ack also run un-gated (pre-Q&A).

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
  status + `signal_ref` (HDF5 pointer), `ecg_plot_ref` (rendered PNG), `hr_history` (HR series for
  the trend query)), `ActionItem`, `Report`, `CareProtocol`/`ProtocolStep`.
- **Edges:** `HAS_DIAGNOSIS`, `CO_MORBID_WITH` (derived from cohort co-occurrence, carries
  confidence/count/window), `PRESENTS`, `HAD_SURGERY`, `PRESCRIBED`, `MANAGES`, `ASSIGNED_TO`
  (Patient→Bed), `IN_UNIT`, `HAD_EVENT`, `AT_BED`, `OF_CONDITION`, `FOLLOWED_BY` (per-patient chain
  ordered by `event_timestamp`), `HAS_ACTION`, `HAS_REPORT`, `APPLIES_TO`, `HAS_STEP`.

**Care protocols** are curated external YAML (`common/protocols/care_protocols.yaml`), matched on
`event_type` + vital conditions + min severity (most-specific-wins, with a default fallback), loaded
into the graph **and** indexed as narrative into the vector store.

**Waveforms & artifacts.** The raw multi-lead ECG lives only in the **HDF5** source (and, briefly, in
`SignalWindow.signals` at the producer — the bus drops it to stay small). So the **producer renders
the ECG strip** (`inference/plotting.py`, primary lead → PNG under `PLOT_DIR`) while the samples are
in hand, and only the **path** (`ecg_plot_ref`) rides the bus and is persisted — the same
materialize-then-reference pattern as the report markdown. Small vital **histories** (HR for now) are
carried on the bus too and persisted as `hr_history`, backing the "how was HR trending?" answer. The
graph stays the structured source of truth; bulky signals stay in HDF5 / the rendered image. *(These
fields populate on fresh `ingest → consume`; older events show "raw ECG is archived" until re-run.)*

### Operational query matrix (the verification target)

Every row is answerable through tested, read-only, parameterized Cypher templates (free
text-to-Cypher is an allowlisted read-only fallback only):

| # | Query | Store |
|---|-------|-------|
| 1 | Critical events last 24h by patient/bed/unit (criticality High+, i.e. call-worthy) | graph |
| 2 | Positive (non-FP) events last *x* min | graph |
| 3 | Event status on a bed (ts, event, FP?, actual condition) | graph |
| 4 | Event analysis report for a patient/bed/unit | graph + vector |
| 5 | Vitals at the time of a specific event | graph (inline snapshot) |
| 6 | Outstanding action items across patients | graph |
| 7 | Care protocol for a bed's last event | hybrid |
| 8 | Patterns: age/gender/co-morbidity/symptom → event type | graph (analytics) |
| 9 | ECG strips for the last event (any type, a named type, or *this patient's*) | graph → PNG artifact |
| 10 | HR & BP trend for last Tachycardia event; HR-history trend for *this patient's* event | graph → plot ref / HR series |

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
| "Show me the **ECG strips** for the patient with the last reported (AFib) event." | T9 (graph → PNG path; producer renders the lead, voice says "an ECG strip is available", companion app shows it) |
| "Show me the **HR and BP trend** for the patient with the last reported Tachycardia event." | T10 (graph → vitals/trend plot ref) |
| "**How was HR trending** at the time of this patient's event?" | HR-history series persisted on the event (POC: HR; RR/SpO2/BP later) — answered as "HR trended from X to Y (rising)…" |
| "What are the **other critical / all events for *this patient***?" | session-patient-scoped (outbound call) — filtered to the bound patient, not all |
| "Show **all patients who have a specific event** (e.g. all AFib)." | graph traversal: `MonitoredEvent {event_type}` → `Patient` (+ optional bed/unit) |

> Operational items (1–6, 9, 10) are judged by **exact-row match**; hybrid/relationship items (7, 8)
> by **groundedness + citations**. Out-of-corpus / unknown-bed / unknown-patient questions are
> expected to **decline**, never fabricate.

Event names in the event-scoped questions (9, 10, "all patients with …") are **parameterized**:
substitute any of the 16 classes — "AFib", "v-tach", "ST elevation", "mobitz 2", "SVT", … — and the
same template runs with a different `event_type` (NL→class resolved by `event_type_from_text`).

The matrix is wired identically for **voice and text** (the voice handler routes through the same
`match_intent`). Spoken queries are normalized first (`kb/graph/spoken.py`) so STT phrasing resolves
like typed: spelled-out acronyms collapse ("A V Block" → "AV Block", "S V T" → "SVT"), number-word
bed labels rebuild ("bed unit one bed oh one" → "Unit1-Bed01"), and spoken counts become digits
("twenty four hours" → 24). So bed/event-type/time-scoped questions work spoken, not just typed.

**Answer style.** Operational (template-matched) questions are answered **deterministically** from the
graph rows — crisp, exact, no LLM in the loop (so a small local model can't pad or garble them), and
with no conversation history or recalled context in the path. Only free-text/hybrid questions go
through the LLM, under a tight instruction to lead with the answer and drop preamble/disclaimers; that
path includes the live conversation history (for follow-ups) but recalled cross-session "past
interactions" only when `EPISODIC_RECALL=true` (off by default). Patient pseudonyms (`PT####`),
clinical terms ("SVT"), and bed/unit labels are preserved through de-identification (presidio NER
would otherwise scrub them as PII).

### Managing & inspecting the KB

Two stores hold an event: the **graph** (Neo4j) is the structured source of truth (the
`MonitoredEvent` + vitals snapshot + links + a `Report` node whose `uri` points at the materialized
`data/reports/<id>.md`); the **vector** store (Qdrant `rmsai_docs`) holds the report *narrative*
(`doc_id=report:<id>`) plus the clinical-protocol corpus, for semantic Q&A.

**Dump what's stored for one event** (graph node + vector chunks + report file, side by side):
```bash
uv run python -m cli.kb_dump --list             # recent event ids
uv run python -m cli.kb_dump <event_id>          # graph + report file + vector chunks
uv run python -m cli.kb_dump <event_id> --json   # raw {graph, vector, report_text}
```
Ad-hoc graph reads use `cli.graph` (templates or read-only Cypher); GUIs: Neo4j Browser
`http://localhost:7474`, Qdrant dashboard `http://localhost:6333/dashboard`.

**Indexing is append-by-default — re-indexing docs no longer wipes event narratives.**
`cli.kb_vector index` upserts (idempotent); pass `--reset` only for a clean rebuild. The
event-writing/serving paths (`cli.consume`, `cli.outbound`, and `build_orchestrator` — which runs on
every text chat **and** every voice call) all **append**, so they preserve the report narratives that
`consume` archives. Append requires a **matching embedder dimension**: the collection is built with
one embedder (hashing=256 / BGE=384); re-index with the same `--embedder`, or `--reset` to rebuild —
a clear error fires on mismatch.

```bash
uv run python -m cli.kb_vector index --dir docs              # append (preserves event reports)
uv run python -m cli.kb_vector index --dir docs --reset      # full rebuild (wipes the collection)
```

> ⚠️ **Destructive ops to know about.** Graph/orchestrator pytest fixtures
> (`tests/test_graph_templates.py`, `tests/test_orchestrator.py`) run against the **live** Neo4j and
> `reset_all()` on teardown — running them wipes ingested data; **re-ingest afterwards**.
> `cli.kb_vector index --reset` and `cli.kb_eval` rebuild the Qdrant collection. After any such wipe,
> re-run the `ingest → consume` flow to repopulate events (and their report narratives).

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
| `CRITICALITY_FP_OVERRIDE_ON_VITALS` | `true` | call even on a confident false-positive ECG (NORMAL_SINUS) when vitals warrant it (MEWS ≥ threshold or deteriorating); overrides the spec-D10 no-call guard |
| `OUTBOUND_ENABLED` / `OUTBOUND_MIN_CRITICALITY` | `false` / `High` | gate which events dial out |
| `OUTBOUND_CALL_NUMBER` / `OUTBOUND_FROM` | — | single hard-configured destination + caller ID |
| `OUTBOUND_MAX_RETRIES` / `OUTBOUND_RETRY_DELAY_S` | `2` / `30` | no-answer retry policy |
| `INBOUND_AUTH_PIN` | shared PIN | verified before any PHI is voiced |
| `AUDIO_WAKE_WORD` | `hey vios` | wake word that gates follow-up *audio* Q&A on a call (text chat is never gated) |
| `AUDIO_WAKE_WINDOW_S` | `30` | seconds the agent stays "awake" after a wake word so audio follow-ups needn't repeat it (audio only — does not affect text chat) |
| `EPISODIC_RECALL` | `false` | condition free-text answers on recalled cross-session past Q&A; off keeps answers grounded in the live KB + current conversation only |
| `STT_LANGUAGE` | `en` | force the STT language (ISO 639-1); blank/`auto` = auto-detect. Stops Whisper/Scribe "hearing" other languages on noise |
| `ECG_PLOT_ENABLED` / `PLOT_DIR` | `true` / `data/plots` | producer renders each event's ECG lead to `{PLOT_DIR}/<event_id>.png` (gitignored); path persisted as `MonitoredEvent.ecg_plot_ref` |
| `DEID_BACKEND` | `auto` | `auto` / `regex` / `presidio` |

**Stubs (POC → production):** `PatientHistory` (synthetic, seeded by `patient_id` → EMR/FHIR) ·
`BedAssignment` (≤25 beds/unit, overflow, clear ops → ADT feed) · `ECGModel` (vendored checkpoint or
deterministic test stub → validated SaMD model on Triton) · `EventStore` (Neo4j → Postgres/
TimescaleDB at scale) · FHIR client (stub → HAPI) · outbound (single number → escalation tree) ·
caller auth (shared PIN → per-user identity/MFA) · audit (JSONL → tamper-evident store).

### Criticality & the outbound-call decision

Criticality (`common/criticality.py`) drives both the persisted `MonitoredEvent.criticality` and the
outbound-call gate, and all of its inputs are configurable (table above). It is computed in layers:

1. **Intrinsic** — `criticality(event_type, mews_risk)` takes the more severe of the arrhythmia
   class and the MEWS risk (so VT/VF/ST_ELEVATION are intrinsically `Critical`).
2. **Configurable escalation** — `event_criticality(event, config)` raises that to **at least
   `High`** when **any** of:
   - the event is **not** the normal baseline (`CRITICALITY_NORMAL_EVENT`, default `NORMAL_SINUS`) —
     i.e. *any* real arrhythmia is at least High;
   - the **MEWS score ≥ `CRITICALITY_MEWS_THRESHOLD`** (default 3);
   - a **vital is deteriorating** (Mann-Kendall trend), when `CRITICALITY_ESCALATE_ON_DETERIORATING`
     is on.

   Escalation only ever raises to `High` — it never lowers an already-`Critical` event.
3. **Call gate** (`should_call`) — dials out when `outbound_enabled` and the criticality is at or
   above `OUTBOUND_MIN_CRITICALITY` (default `High`). The event is **always persisted**; the gate
   only governs the call.

**False-positive override (vitals beat the rhythm).** A confident `NORMAL_SINUS`
(≥ `FP_SUPPRESS_MIN_CONFIDENCE`) is a false positive and normally does **not** call (spec D10). When
`CRITICALITY_FP_OVERRIDE_ON_VITALS` is on (default), a **vitals-driven escalation** (MEWS ≥ threshold
or deteriorating) **overrides** that guard — the patient is deteriorating regardless of the rhythm,
so the call still fires. This case is made explicit at three layers so it is never confusing:

- **decision reason** → `fp_override (MEWS 5 >= threshold 3)` / `fp_override (vitals deteriorating)`
  (also written to the audit log);
- **console** → `[consume] FALSE-POSITIVE OVERRIDE: ECG classified NORMAL_SINUS (false positive),
  but the patient's vitals warrant a call … — vitals/MEWS-driven escalation, not the rhythm.`;
- **spoken alert** → appends *"Note: the ECG rhythm is classified as normal sinus, so this alert is
  driven by the patient's vitals (…), not the rhythm."*

Set `CRITICALITY_FP_OVERRIDE_ON_VITALS=false` to revert to the strict spec-D10 behaviour
(NORMAL_SINUS ⇒ never call).

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

**Swappable speech backends.** STT/TTS sit behind `STTAdapter`/`TTSAdapter` (`voice/adapters.py`),
selected by `STT_BACKEND` / `TTS_BACKEND`: self-hosted **whisper**/**piper** (default), or cloud
**elevenlabs** (STT "Scribe" + TTS; stdlib HTTP, no extra dep) for **accuracy/latency benchmarking**.
Set `ELEVENLABS_API_KEY` (+ optional `ELEVENLABS_VOICE_ID` / `ELEVENLABS_TTS_MODEL` /
`ELEVENLABS_STT_MODEL`). Compare backends + latency offline with
`uv run python -m cli.speech_check --tts elevenlabs --stt elevenlabs` (TTS → STT round-trip, prints
per-leg ms).

PHI handling differs by direction (the two cloud legs are **not** symmetric):

- **Cloud TTS is PHI-guarded.** Every spoken string (greeting, alert, answers) is run through the
  configured de-identifier (`DEID_BACKEND`: Presidio/regex) by `DeidentifyingTTS` **before** the
  text leaves the host — on top of pseudonym-by-construction. So no name/SSN/etc. reaches the
  provider; cloud TTS is safe for the de-identified clinical text this system produces.
- **Cloud STT cannot be pre-redacted.** It sends **raw caller audio** to be transcribed, so there is
  nothing to de-identify first (de-id needs text). If a clinician *speaks* an identifier it reaches
  the provider. Therefore `STT_BACKEND=elevenlabs` is **synthetic-speech only, never real PHI**
  (hard rules #4/#5). The worker prints a warning. The self-hosted whisper path keeps STT on-box.

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
# → join the printed rmsai-outbound-<event_id> room (playground or meet.livekit.io)
# → PIN → hear THIS event's alert → Q&A (see below) → say "acknowledge" (flips event status)
```

> Each event gets its **own** room (`rmsai-outbound-<event_id>`) and a **fresh** join token, both
> printed by `cli.consume`. The static `rmsai-outbound` name is for SIP phone dialing only. The
> worker joins when you join (auto-dispatch); a worker started before a code change won't pick it up
> for an already-live room — restart the worker **and** place a new call.

**Talking vs typing during a call (modality-matched replies).** Once past the PIN and the spoken
alert, you can interact two ways and the response matches the input modality:

- **Speak** — follow-up *audio* Q&A is gated by a **wake word** (`AUDIO_WAKE_WORD`, default
  `"hey vios"`) so room noise and Whisper hallucinations don't trigger replies. Say e.g.
  *"hey vios, what were the vitals at the time of the event?"* → **spoken** answer. The agent then
  stays "awake" for `AUDIO_WAKE_WINDOW_S` (default 30 s), so immediate follow-ups don't need to
  repeat the wake word. Audio with no wake word (and outside the window) is silently ignored.
- **Type** — a message in the LiveKit **chat box** gets a **text-only** reply on the chat channel
  (no TTS/audio). Typed turns are never wake-word gated.

The PIN entry, the spoken alert, and the verbal **"acknowledge"** run *before* the Q&A phase and are
never wake-word gated. Patient-scoped questions resolve "the event" to that patient's most recent
`MonitoredEvent` (e.g. vitals → `T5`).

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
uv run python -m cli.kb_dump <event_id>    # dump one event: graph node + report file + vector chunks
uv run python -m cli.kb_eval               # vector vs hybrid over the gold question set
uv run python -m cli.memory demo           # working + episodic + semantic memory round-trip
uv run python -m cli.speech_check          # offline Piper TTS → Whisper STT round-trip
uv run python -m cli.text_chat             # inbound query by text (PIN-gated, KB-grounded)
uv run python -m cli.voice                 # offline typed-text voice demo (no audio hardware)
```
