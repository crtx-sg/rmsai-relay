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

> For a plain-language walkthrough of the end-to-end control flow, data flow, and every model
> (ECG classifier, LLM, STT/TTS, embeddings), see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

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
  `PatientHistory` stubs). `event_type` ∈ the 16-name `ecgtranscnn` vocabulary (a given
  checkpoint's head is a subset — the real-ECG `real_v2` model predicts 13 of them);
  `NORMAL_SINUS` ⇒ FP.

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
| Infra | Docker Compose (`infra/docker-compose.yml`) — **everything runs as a service**: the backing stores (neo4j, qdrant, redis, livekit; profiled: mosquitto, model-server/ollama, hapi-fhir) *and* the app itself (consumer, voice-worker, gateway) from one shared image (`infra/Dockerfile`) with the source bind-mounted |

**No SQL DB in the POC.** Five stores: Neo4j (relationships + operational event log, behind an
`EventStore` repository interface so it can migrate to Postgres/TimescaleDB later), Qdrant (text +
embeddings), Redis (working memory + bus), HDF5 (waveforms), object/file store (plots, reports).

### Repository layout

`common/` contracts + config + de-id + protocols · `ingest/` HDF5 + MQTT readers · `inference/`
model + vitals + serialize · `kb/{vector,graph,hybrid}` retrieval · `memory/` working/episodic tiers
· `orchestrator/` turn loop + outbound flow + bus consumer · `voice/` SIP/LiveKit + handlers +
STT/TTS · `emr/` FHIR · `app/`+`live/` companion app & live media (Phase 9, planned) · `cli/`
entrypoints · `infra/` compose + app `Dockerfile` · `external/ecgtranscnn/` vendored model+simulator
(gitignored) · `data/{synthetic,fixtures}` · `docs/` clinical corpus (+ `docs/samples/` upload
fixtures, excluded from it).

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
one embedder (hashing=256 / BGE=384), so those paths default `--embedder` to `EMBEDDER` from `.env`;
re-index with the same embedder, or `--reset` to rebuild — a clear error fires on mismatch.

```bash
uv run python -m cli.kb_vector index --dir docs              # append (preserves event reports)
uv run python -m cli.kb_vector index --dir docs --reset      # full rebuild (wipes the collection)
```

**The corpus is `docs/*.md` plus the managed upload folder.** `chunk_dir` globs `docs/` **non-
recursively for `*.md`**, so every markdown file dropped there becomes retrievable clinical
evidence — project/engineering docs belong in `docs/project/` and upload test fixtures in
`docs/samples/`, both of which the glob excludes. Non-markdown files are never picked up by the
glob: a PDF must go through `cli.kb_upload`. Documents added that way are copied to
`KB_UPLOAD_DIR` (`data/kb_uploads/`, gitignored) and re-indexed by every rebuild, so they are part
of the corpus rather than a one-off write. See [§3a](#3a-upload-protocols--sops--checklists-to-the-kb)
for the upload + verification commands.

**How a question is answered** — three paths, in order:

| path | when | answer |
|---|---|---|
| graph template | `match_intent` regex hits | exact patient data from Cypher, no LLM |
| LLM-routed template | regex misses, question looks operational, `KB_LLM_ROUTER=true` | same, template chosen by the model from a fixed menu |
| hybrid retrieval | everything else | cited passages (vector) + relationships (graph), answered by the LLM |

`uv run python -m cli.kb_route --llm "<question>"` shows which path a question takes and why.

**Relevance gate.** A hybrid answer is only given when retrieval vouches for itself: a graph
relationship, semantic similarity ≥ `KB_MIN_RELEVANCE` (0.60), **or** word overlap ≥ 0.18. The two
signals fail in opposite directions — semantic similarity survives rewording (calibrated on this
corpus: on-topic 0.61–0.83, off-topic 0.35–0.54, but only under `EMBEDDER=bge`; hashing scores the
two alike), while word overlap is what carries the hashing embedder. When neither vouches, the turn
declines and logs **why**, naming both numbers, both thresholds, and the top passage:

```
[orchestrator] declined: 4 passage(s) retrieved, but neither relevance signal vouched for them:
semantic similarity 0.17 < 0.60 AND word overlap 0.00 < 0.18 (top: vt_vf.md#Post-resuscitation)…
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
| `OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE` | `0.60` | a non-normal (arrhythmia) event is only asserted *as a rhythm* if confidence is at/above this. Below it: withheld when the vitals are calm, or re-based as a **vitals-driven alert** (rhythm marked unconfirmed) when they aren't. Vitals never raise the bar |
| `OUTBOUND_CALL_NUMBER` / `OUTBOUND_FROM` | — | single hard-configured destination + caller ID |
| `OUTBOUND_MAX_RETRIES` / `OUTBOUND_RETRY_DELAY_S` | `2` / `30` | no-answer retry policy |
| `INBOUND_AUTH_PIN` | shared PIN | verified before any PHI is voiced |
| `AUDIO_WAKE_WORD` | `hey vios` | wake word that gates follow-up *audio* Q&A on a call (text chat is never gated) |
| `AUDIO_WAKE_WINDOW_S` | `30` | seconds the agent stays "awake" after a wake word so audio follow-ups needn't repeat it (audio only — does not affect text chat) |
| `AUDIO_WAKE_REQUIRED` | `true` | require the wake word to open a follow-up audio turn (SIP/playground Q&A). Set `false` to answer **every** authenticated audio turn — the escape hatch when STT mishears the out-of-vocab brand word. The companion app already bypasses the gate (it controls the mic) |
| `INBOX_SPEAK_ON_SELECT` | `true` | companion app: selecting a worklist row speaks that event's stored report summary aloud (in addition to scoping chat). Spoken text is the `Report.summary` — no model call |
| `LIVEKIT_REDISPATCH_ON_START` | `true` | on worker startup, auto re-dispatch the agent into live `rmsai-inbox-*` rooms that lost their agent (e.g. after a worker restart), so the app doesn't need a re-login. On-demand equivalent: `cli.dispatch` |
| `LIVEKIT_WORKER_HTTP_PORT` | `8081` | port for livekit-agents' health-check HTTP server. Only matters when two workers run side by side — the Docker `voice-worker` service defaults it to `8091`, because host networking would otherwise collide with a worker run by hand (`[errno 98] address already in use`) |
| `GATEWAY_PORT` | `8080` | host port for the containerized gateway. Move it if `hapi-fhir` (`--profile emr`) or a hand-run gateway already holds 8080 |
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

**Arrhythmia confidence gate (the model must be sure — or say it isn't).** A **non-normal
(arrhythmia)** prediction is only asserted *as a rhythm* when the model's confidence in it is at/above
`OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE` (default `0.60`). Below that, what happens next depends on the
patient, not on the classifier:

- **vitals unremarkable** → withheld (`low_confidence_arrhythmia (45% < 60%)`). The event is still
  persisted and answerable in chat; only the alert is suppressed.
- **vitals warrant attention** (MEWS ≥ threshold or a deteriorating trend) → the alert still goes out,
  but **re-based on the vitals** (`vitals_alert (MEWS 4 >= threshold 3)`). The worklist row leads with
  *Vitals alert* and the named vital, the rhythm is demoted to `unconfirmed: <type>`, and the spoken
  call alert says "Possible …, treat the rhythm as unconfirmed".

Vitals never raise the confidence bar — they cannot make an uncertain classification true — they only
change what the alert **claims**. `common.criticality.alert_basis` decides that once, and every
surface leads with it. Rows also carry their confidence, flagged below `LOW_CONFIDENCE_CAVEAT`.
See the flowchart in [ARCHITECTURE.md](ARCHITECTURE.md) or [`docs/alert-gate.html`](docs/alert-gate.html).

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

Two ways to run it. **Docker** brings the whole relay up in one command and needs no host Python;
**host** installs the environment locally, which is the better loop for editing and running tests.
They share the same `.env` and the same datastores, so you can mix them.

### A. Everything in Docker (recommended for a demo)

```bash
# 1. Vendor the ECG model + simulator (gitignored; baked into the image, so clone it FIRST)
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn

# 2. Configure
cp .env.example .env      # then set LIVEKIT_API_KEY / LIVEKIT_API_SECRET, HOSPITAL_ID, …

# 3. Build the shared app image (once; ~5-10 min for torch + whisper + presidio)
make docker-build

# 4. Bring up every service
make docker-up
```

That starts **redis, neo4j, qdrant, livekit** and the three application services — **consumer**
(bus → persist → dispatch), **voice-worker** (the LiveKit agent), and **gateway** (the companion
app on `http://localhost:8080/`). `make docker-ps` shows the state, `make docker-logs` tails the
three app services.

The **source is bind-mounted**, so a code edit needs only `make docker-restart` — no rebuild.
Rebuild (`make docker-build`) only when `pyproject.toml`, `uv.lock`, or the vendored package
changes.

Run any CLI harness in the same image, with no host Python:

```bash
docker compose -f infra/docker-compose.yml run --rm tools \
    python -m cli.ingest --file data/inference/<f>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt --emit bus
make docker-shell        # or an interactive shell in that image
```

> **App services use `network_mode: host`.** That is deliberate: it makes every `localhost` URL in
> `.env` resolve the same inside a container as on the host, keeps an Ollama running on the *host*
> reachable, and — the reason that matters most — lets LiveKit's advertised `rtc.node_ip: 127.0.0.1`
> mean the same thing to the voice worker as to the browser. On a bridge network the worker would
> receive that candidate, try to connect to itself, and audio would never flow even though
> signaling succeeded. The tradeoff is no port isolation, and Linux-only portability. See the header
> of `infra/docker-compose.yml`.

Two services stay behind profiles because they collide with this setup: `--profile llm`
(`model-server`, containerized Ollama — **skip it if `ollama serve` already runs on the host**;
both bind 11434) and `--profile emr` (`hapi-fhir` — binds 8080, same as the gateway; move the
gateway with `GATEWAY_PORT`). `--profile telemetry` adds mosquitto. The legacy `--profile later`
still selects all of them.

### B. On the host

```bash
# 1. Vendor the ECG model + simulator (gitignored; see external/ecgtranscnn/PLACEHOLDER.md)
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn

# 2. Install everything (recommended for a demo). `make setup-all` = all extras
#    (rag, deid, voice, livekit, app) + the vendored ecgtranscnn editable + the spaCy model.
make setup-all
#    Lean alternative (core + dev only): `make setup`.
#    Re-run `make external` after ANY hand-run `uv sync` — the vendored package is gitignored, so
#    `uv sync` uninstalls it and it must be reinstalled editable.

# 3. Bring up the backing stores only (leave the app services to your terminals)
make stores-up            # = docker compose … up -d redis neo4j qdrant livekit
make stores-check         # which ones are actually reachable?

# 4. Run tests
uv run pytest
```

> **The stores run in Docker even in this mode.** Running the CLIs with `uv run` does *not* make
> them self-contained — `cli.ingest --emit bus` still needs Redis, `cli.consume` needs Redis + Neo4j
> + Qdrant, and the app needs LiveKit. A `docker compose down` removes them all, and the next CLI
> run fails with `Error 111 connecting to localhost:6379`. `make stores-up` is the fix;
> `make stores-check` tells you which one is missing:
>
> ```
> redis   : up
> neo4j   : DOWN  -> make stores-up
> qdrant  : up
> livekit : up
> ```
>
> Use `make stores-up`, **not** `make docker-up`, when you drive the app by hand — the latter also
> starts consumer/voice-worker/gateway, which collide with host-run copies on 8080/8081.

### Make targets

| Target | What it does |
|---|---|
| `make setup` | `uv sync --extra dev` + `make external` (core + dev only) |
| `make setup-all` | all extras (rag, deid, voice, livekit, app) + `make external` + spaCy `en_core_web_sm` |
| `make external` | (re)install the vendored `external/ecgtranscnn` editable — run after any manual `uv sync` |
| `make stores-up` / `make stores-down` | start / stop **only** the backing stores (redis, neo4j, qdrant, livekit) — the target to use when you run the CLIs on the host |
| `make stores-check` | which backing stores are reachable, and what to run if one is down |
| `make docker-build` | build the shared app image (`infra/Dockerfile`) |
| `make docker-up` / `make docker-down` | start / stop every service |
| `make docker-restart` | restart the three app services to pick up source edits (no rebuild) |
| `make docker-logs` / `make docker-ps` | tail the app services / show container state |
| `make docker-shell` | interactive shell in the app image with the repo mounted |
| `make test` / `make lint` | `uv run pytest -q` / linters |

### Optional extras (à la carte, if you skipped `make setup-all`)

```bash
uv sync --extra rag                                   # real BGE embeddings + reranker
uv sync --extra deid && uv run python -m spacy download en_core_web_sm   # Presidio de-id
uv sync --extra voice                                 # faster-whisper STT + Piper TTS
uv sync --extra livekit                               # LiveKit agent worker + SIP/WebRTC
# NOTE: any bare `uv sync` uninstalls the vendored ecgtranscnn — follow with `make external`.
```

`.env` keys that matter: `REDIS_URL`, `NEO4J_*`, `QDRANT_URL`, `DEID_BACKEND`, `STT_BACKEND`/
`TTS_BACKEND`, `INBOUND_AUTH_PIN`, `AUDIO_WAKE_REQUIRED` (wake-word gate on/off),
`INBOX_SPEAK_ON_SELECT` (voice the event on worklist select), and for live calls `LIVEKIT_URL` /
`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` plus `OUTBOUND_ENABLED=true`.

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

## Operations runbook

The whole lifecycle in order — bring the stack up, load the model, initialize the KB, drive it,
inspect it, measure it, shut it down. Every command here is Docker-first; the host equivalent of
any `$RMSAI` line is the same `python -m …` under `uv run`.

Define this once per shell — it runs any CLI harness in the app image with the repo mounted, so no
host Python is needed:

```bash
export RMSAI="docker compose -f infra/docker-compose.yml run --rm tools python -m"
$RMSAI cli.kb_dump --list          # …and so on for every cli.* below
```

### 1. Prerequisites and the vendored ECG model

The `ECG_TransConv` classifier and the synthetic-data simulator are **vendored, not vendorable by
pip** — the clone carries the `scripts/` simulators and `models/` checkpoints that a wheel would
not. It is gitignored but **baked into the image**, so clone it *before* the first build:

```bash
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn   # pinned: 0bc646da
cp .env.example .env    # then set LIVEKIT_API_KEY / LIVEKIT_API_SECRET / HOSPITAL_ID
```

Check the checkpoints arrived — weights may be Git-LFS and a plain clone can omit them:

```bash
ls external/ecgtranscnn/models/*/best_model.pt
# external/ecgtranscnn/models/avblock_fix/best_model.pt
# external/ecgtranscnn/models/noise_robust/best_model.pt
```

**Without weights the pipeline still runs** — `ECGModel` falls back to the deterministic stub, so
every phase and test works; only the predictions are synthetic. With weights, pass one as
`--checkpoint` to `cli.ingest` / `cli.outbound` (below). `noise_robust/best_model.pt` is the
general-purpose one.

Generate synthetic HDF5 to feed the pipeline. The generator's `--output-dir` defaults to
`data/inference` **relative to the current directory**, so run it from the repo root (or pass the
flag) — otherwise the files land somewhere you then can't find, e.g. under
`external/ecgtranscnn/data/inference/` if you `cd` in there first:

```bash
# a script rather than a module, so it does not use $RMSAI
docker compose -f infra/docker-compose.yml run --rm tools \
    python external/ecgtranscnn/scripts/generate_inference_data.py --output-dir data/inference
ls data/inference/*.h5
```

Useful flags: `--num-files` / `--events-per-file` (default 3 × 5), `--conditions
ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1` to weight the mix, and `--noise-level {low,medium,high,mixed}`.
`data/` is gitignored and bind-mounted into every container, so these files are visible to both the
host CLIs and the `tools` service at the same path.

### 2. Build and start

```bash
make docker-build      # once; ~5-10 min (torch, faster-whisper, presidio, spaCy)
make docker-up         # redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway
make docker-ps
```

Wait for all three app services to report ready — cold start loads BGE/Whisper/the orchestrator:

```bash
docker compose -f infra/docker-compose.yml logs consumer     | grep "\[consume\] group="
docker compose -f infra/docker-compose.yml logs voice-worker | grep "registered worker"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/     # 200
```

Expected steady state:

```
consumer       Up  (blocking on XREADGROUP)
gateway        Up (healthy)   → http://localhost:8080/
voice-worker   Up  (registered as rmsai-agent, health server on 8091)
livekit neo4j qdrant redis   Up (healthy)
```

> **Before `make docker-up`, stop any host-run `cli.gateway` / `cli.voice_worker`.** The app
> services use host networking, so a hand-run gateway holds 8080 and a hand-run worker holds 8081 —
> the containers then fail to bind. Two workers registered under the same `LIVEKIT_AGENT_NAME` also
> split dispatches between them.

### 3. Initialize the knowledge base

Both stores start empty. The **graph** needs its schema, care protocols, and document entities; the
**vector** store needs the clinical corpus. `cli.consume` runs `migrate` itself on startup, so
strictly only the middle three are manual:

```bash
$RMSAI cli.graph migrate                     # constraints + indexes   -> {"migrated": true}
$RMSAI cli.graph protocols                   # care protocols          -> {"protocols_loaded": 2}
$RMSAI cli.graph extract --dir docs          # doc entities onto shared nodes
#   -> {"chunks": 13, "guidelines": 13, "conditions": 5, "treatments": 9}
$RMSAI cli.kb_vector --embedder bge index --dir docs    # clinical corpus -> Qdrant
```

Optionally seed a synthetic patient cohort (demographics, co-morbidities, symptoms) so the
pattern/co-morbidity queries have something to traverse. Patients are auto-created on first event
otherwise (G8):

```bash
$RMSAI cli.graph ingest --patients PT1000 PT1001 PT1002
```

**The embedder must match the collection** (hashing=256-dim / BGE=384-dim). Every KB path defaults
`--embedder` to `EMBEDDER` from `.env`; `cli.kb_vector` is the one exception (it defaults to
`auto`), so pass it explicitly there. See [§3a](#3a-upload-protocols--sops--checklists-to-the-kb).

### 4. Drive the application

Publish classified events onto the bus; the running `consumer` container picks them up, persists to
both stores, and dispatches per the criticality gate:

```bash
$RMSAI cli.ingest \
    --file data/inference/PT6580_2026-06.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt --emit bus
```

```
{"published": "…", "patient": "PT6580", "event_type": "VENTRICULAR_TACHYCARDIA",
 "confidence": 1.0, "criticality": "Critical", "mews": 6, …}
```

Watch it land:

```bash
docker compose -f infra/docker-compose.yml logs -f consumer
# [consume] received event daf16c2d… type=VENTRICULAR_TACHYCARDIA conf=1.00 patient=PT6580
# [consume] persisted MonitoredEvent … -> Neo4j graph
# [consume] archived report narrative -> Qdrant vector store
# [consume] dispatch=app: pushed inbox event … -> rmsai-inbox-h1
```

Then **use it**: open `http://localhost:8080/`, enter the PIN (`INBOUND_AUTH_PIN`, default `1234`),
and the worklist renders live. Selecting a row scopes chat to that event and speaks its report
summary (`INBOX_SPEAK_ON_SELECT`). Ask questions by typing or by voice. For the phone/WebRTC paths
and the on-demand call, see [End-to-end testing](#end-to-end-testing) §5–6.

Query the KB directly without the app:

```bash
$RMSAI cli.kb "which conditions are co-morbid with atrial fibrillation"   # hybrid: vector + graph
$RMSAI cli.text_chat                                                       # PIN-gated text console
```

### 5. Load documents into the KB

Protocols, SOPs, and checklists — PDF, markdown, or text — so questions are answered from *your*
documents with a citation instead of declined:

```bash
$RMSAI cli.kb_upload --file protocols/af_sop.pdf --dry-run   # preview the chunk plan
$RMSAI cli.kb_upload --file protocols/af_sop.pdf
$RMSAI cli.kb_upload --dir protocols/ --glob '*.pdf' --extract   # + graph entities

# a sample SOP ships with the repo, deliberately outside the auto-indexed corpus
$RMSAI cli.kb_upload --file docs/samples/critical_alarm_sop.md
```

Uploads are idempotent, copied into `KB_UPLOAD_DIR` so a rebuild re-indexes them, and PDF pages are
cited individually. Full detail + verification commands:
[§3a](#3a-upload-protocols--sops--checklists-to-the-kb).

### 6. Test the application

```bash
$RMSAI pytest -q                        # full suite (needs the stores up)
$RMSAI pytest -q -m "not infra"         # offline only — no containers required
$RMSAI pytest -q tests/test_criticality.py -k fp_override    # one file / one test
make test                               # host equivalent (uv run pytest -q)
```

Then the end-to-end smoke test — generate, classify, publish, consume, alert, acknowledge:

```bash
$RMSAI cli.ingest --file data/inference/<f>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt --emit bus
docker compose -f infra/docker-compose.yml logs --tail=40 consumer
```

Expect critical events (AFib/VT, High/Critical) persisted **and** dispatched; `NORMAL_SINUS`/Low
persisted but skipped with a printed reason (`below_threshold`, `low_confidence_arrhythmia`,
`vitals_alert`, `fp_override`). Per-subsystem harnesses are listed under
[Other CLI harnesses](#other-cli-harnesses-per-subsystem).

> ⚠️ **`tests/test_graph_templates.py`, `tests/test_orchestrator.py`, and `cli.kb_eval` call
> `reset_all()` on the LIVE Neo4j.** Running the full suite wipes ingested events — that is why
> `cli.kb_dump --list` can suddenly return `[]`. Recover by re-running step 3 then step 4.

### 7. Debug and inspect

**Dump everything stored for one event** — the graph node, the report file, and the vector chunks,
side by side. This is the first thing to reach for when an answer looks wrong:

```bash
$RMSAI cli.kb_dump --list                    # recent event ids to pick from
$RMSAI cli.kb_dump <event_id>
$RMSAI cli.kb_dump <event_id> --json         # raw {graph, vector, report_text}
```

```
=== EVENT daf16c2d-51fb-4cbe-a4ef-dea59502e56b ===
GRAPH (Neo4j)
  patient    : PT6580        bed : Unit1-Bed01
  event_type : VENTRICULAR_TACHYCARDIA | criticality Critical | status reported | FP False
  confidence : 0.9999  | MEWS risk High
  vitals     : HR 175.0, BP 175.0/93.0, SpO2 93.0, RR 24.0, Temp 99.0
  actions    : MEWS 6 (High) — escalate care
  plots      : ecg=data/plots/daf16c2d….png
  report     : report:daf16c2d…  (index_status indexed)
REPORT FILE (markdown) …
```

**Graph queries** — the operational templates, or a natural-language lookup that shows which
template it resolved to:

```bash
$RMSAI cli.graph template outstanding_action_items
$RMSAI cli.graph lookup "critical events in the last 24 hours"
#   -> {"mode": "template", "template": "critical_events_since", "rows": [...]}
```

**Why did a question take that path?** Regex template vs LLM router vs document retrieval:

```bash
$RMSAI cli.kb_route "what were the vitals at the event"
# [route] regex -> (no match)
# [route] looks operational? True (the LLM router is eligible)
# [route] pass --llm to try the LLM router
```

**Vector store** — what's actually indexed, and does the right passage win:

```bash
$RMSAI cli.kb_vector --embedder bge retrieve "escalation checklist for a critical alarm" -k 3
curl -s -X POST http://localhost:6333/collections/rmsai_docs/points/scroll \
  -H 'Content-Type: application/json' -d '{"limit":500,"with_payload":true}' \
| python3 -c "import sys,json,collections; c=collections.Counter(p['payload'].get('doc_id') for p in json.load(sys.stdin)['result']['points']); [print(f'{v:4d}  {k}') for k,v in sorted(c.items())]"
```

**In-app chat / push-to-talk not responding?** Probe the room without a browser — this splits a
worker fault from a browser fault (stale `app.js`, handler never fired):

```bash
$RMSAI cli.inbox_probe --ptt
$RMSAI cli.inbox_probe --select <event_id> --say "what were the vitals at the event?"
$RMSAI cli.dispatch --all-inbox       # re-wire live inbox rooms that lost their agent
```

**Logs, audit trail, and GUIs:**

```bash
make docker-logs                                    # all three app services, follow
docker compose -f infra/docker-compose.yml logs -f voice-worker
tail -f data/audit.jsonl                            # {ts, actor, action, subject, outcome}
```

Neo4j Browser `http://localhost:7474` · Qdrant dashboard `http://localhost:6333/dashboard` ·
companion app `http://localhost:8080/`.

**Common failures and what they actually mean:**

| Symptom | Cause |
|---|---|
| `Error 111 connecting to localhost:6379` | the backing stores aren't running — `make stores-check`, then `make stores-up`. Running the CLIs under `uv` does not make them self-contained |
| `[errno 98] address already in use` on 8080/8081 | a host-run gateway/worker is still up — host networking shares the port space |
| worker retry-loops on `:7880` | the LiveKit **server** isn't running, not a worker bug |
| `collection … has vector dim 384, but embedder … 256` | `--embedder` doesn't match what built the collection |
| `cli.kb_dump --list` returns `[]` | the graph was wiped (full pytest run / `cli.kb_eval`) — redo steps 3–4 |
| worklist empty, chat silent | worker not dispatched into the room → `cli.dispatch --all-inbox`; or a cached `app.js` (check the on-screen build tag) |
| answer declines on an in-corpus question | relevance gate — the log names both signals and both thresholds |

### 8. Performance

**Retrieval quality + cost + latency**, vector vs hybrid over the gold question set — the go/no-go
on hybrid. Reports correctness, citation grounding, context-token cost, and per-mode latency:

```bash
$RMSAI cli.kb_eval                     # ⚠️ resets the live Neo4j (seeds its own eval cohort)
$RMSAI cli.kb_eval --json
```

**Speech latency per leg** — TTS → STT round-trip with no audio hardware, the way to compare a
self-hosted backend against a cloud one:

```bash
$RMSAI cli.speech_check                                    # whisper + piper (self-hosted)
$RMSAI cli.speech_check --tts elevenlabs --stt elevenlabs  # cloud (synthetic text only)
```

**Container resource use:**

```bash
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'
# infra-consumer-1      0.10%   262.5MiB / 15.35GiB
# infra-voice-worker-1  1.25%    72.8MiB / 15.35GiB   (idle; a live job forks a ~600MB child)
# infra-gateway-1       0.27%    35.0MiB / 15.35GiB
```

The voice worker forks a **job process per room**; livekit-agents logs a memory warning above
500 MB (advisory) and kills a job process that stops answering health pings — which is what a slow
cold start inside a contended container looks like. Per-turn spans (`common/tracing.py`) are
recorded on every `TurnResult.trace` for step-level latency.

Knobs that move the needle: `WHISPER_MODEL` (`tiny.en` ≫ faster than `base.en`), `EMBEDDER`
(`hashing` skips the BGE load entirely), `LLM_PROVIDER=echo` (no model call at all),
`KB_LLM_ROUTER=false` (default — it adds a model call in front of a Cypher lookup, ~6s on
llama3.2:3b), and `EPISODIC_RECALL=false` (default).

### 9. Shut down

```bash
make docker-down          # stop + remove containers; named volumes survive
```

Data survives a `down`: Neo4j (`neo4j_data`), Qdrant (`qdrant_data`), the model cache (`hf_cache`),
and everything under `./data` (reports, plots, uploads, audit log) are volumes or bind mounts.
Redis is in-memory by design (`--save ""`), so the **bus backlog and working memory do not survive**.

```bash
docker compose -f infra/docker-compose.yml down -v      # ⚠️ ALSO deletes graph + vectors + model cache
docker compose -f infra/docker-compose.yml stop consumer   # stop one service
docker compose -f infra/docker-compose.yml restart voice-worker
```

After `down -v` you are back to step 3 — re-initialize the KB, then re-ingest.

---

## End-to-end testing

### 0. Demo bring-up sequence

**All in Docker** — one command brings up the stores, LiveKit, and the three app services:

```bash
make docker-up
make docker-ps        # every service should be Up (gateway: Up (healthy))
make docker-logs      # tail consumer + voice-worker + gateway

# drive a scenario: produce an event into the bus (steps 2-4 below run the same way)
docker compose -f infra/docker-compose.yml run --rm tools \
    python -m cli.ingest --file data/inference/<f>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt --emit bus
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

> Start order matters for in-app chat: the **worker must be running before** a room is created, or
> the `/session` dispatch won't reach it. If you **restart the worker** while the app is connected,
> the existing room is left agent-less (chat/select hit an empty room → no reply, no speech). The
> worker auto re-dispatches into live `rmsai-inbox-*` rooms on startup (`LIVEKIT_REDISPATCH_ON_START`,
> default on); to re-wire immediately without waiting or re-logging in:
>
> ```bash
> uv run python -m cli.dispatch --all-inbox        # every live inbox room lacking an agent
> uv run python -m cli.dispatch --room rmsai-inbox-h1   # one specific room
> ```
> The agent joins a few seconds later (cold-start loads STT/TTS/orchestrator).

### 1. Tests (offline, no infra)

```bash
uv run pytest -q                       # full suite
uv run pytest -q -m "not infra"        # skip infra-dependent tests
```

### 2. Generate synthetic events (vendored simulator)

```bash
# writes HDF5 under <cwd>/data/inference/ — run from the repo root, or pass --output-dir
uv run python external/ecgtranscnn/scripts/generate_inference_data.py --output-dir data/inference
```

### 3. Direct relay path (HDF5 → call, bypasses the bus)

```bash
uv run python -m cli.outbound \
    --file data/inference/<file>.h5 \
    --checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt
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

Uploads are incremental and idempotent (chunk ids are content-addressed), so re-uploading an
unchanged file is a no-op and an edited one replaces only what changed. PDF pages are cited
individually — an answer from `af_sop.pdf#page 3` tells the clinician which page to turn to — and a
scanned/image PDF is rejected with that diagnosis rather than silently indexed as empty.

**The embedder must match the collection.** Vectors are only comparable within one embedder
(hashing=256-dim / BGE=384-dim), so every KB path defaults `--embedder` to **`EMBEDDER`** from
`.env` — set it once and `kb_upload`, `consume`, `outbound`, `text_chat`, and the voice worker all
agree. A mismatch is refused with the fix rather than corrupting the store. (`cli.kb_vector` is the
exception: it defaults to `auto`, which prefers BGE and falls back to hashing offline — pass
`--embedder` explicitly there if `EMBEDDER=hashing`.)

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

Rings `OUTBOUND_CALL_NUMBER` because you asked it to, not because something happened. Each call gets
its own room (`rmsai-call-<id>`); the agent is dispatched there **before** the dial, and with no
alert staged in that room the worker runs the PIN-gated Q&A handler — so the callee authenticates,
then asks grounded questions.

```bash
uv run python -m cli.call                    # simulated: whole path, no telephony
uv run python -m cli.call --caller livekit   # real SIP via LIVEKIT_SIP_TRUNK_ID
uv run python -m cli.call --caller livekit --to +15551234567 --no-dispatch   # trunk test only
```

Needs an outbound trunk (`LIVEKIT_SIP_TRUNK_ID`) and a caller ID (`OUTBOUND_FROM`) — most carriers
reject a call presenting no valid from-number. Exit code is 0 answered / 1 no-answer / 2 invalid.

### 4. Bus path: producer → Redis Stream → consumer

```bash
# PRODUCER — classify + publish to rmsai.events
uv run python -m cli.ingest \
    --file data/inference/<file>.h5 \
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

**Outbound (event-driven, carries the event context):**
```bash
# produce an event (step 4 producer), then stage the alert + print a join token
uv run python -m cli.consume --channel voice --caller livekit --transport webrtc --once
# → join the printed rmsai-outbound-<event_id> room (playground or meet.livekit.io)
# → PIN → hear THIS event's alert → Q&A (see below) → say "acknowledge" (flips event status)
```

> Each event gets its **own** room (`rmsai-outbound-<event_id>`) and a **fresh** join token, both
> printed by `cli.consume` — this holds for **both** transports (SIP and WebRTC), so the room carries
> that event's staged alert. The static `rmsai-outbound` name (`LIVEKIT_SIP_ROOM`) is only the
> *default* for the standalone `cli.outbound` path (one call at a time), not the bus consumer. The
> worker joins when the room is dispatched; a worker started before a code change won't pick it up for
> an already-live room — restart the worker **and** place a new call.

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
uv run python -m cli.ingest --file data/inference/<file>.h5 \
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
uv run python -m cli.kb_upload --dir protocols/    # add SOPs/guidelines/checklists (PDF + markdown)
uv run python -m cli.kb_route --llm "..."  # which path a question takes (regex / LLM / documents)
uv run python -m cli.call --caller livekit # ring the on-call number on demand (no event needed)
uv run python -m cli.inbox_probe --ptt     # probe in-app chat/PTT without the browser
uv run python -m cli.graph migrate         # graph schema migrate / seed
uv run python -m cli.kb "..."              # hybrid (vector + graph) KB query
uv run python -m cli.kb_dump <event_id>    # dump one event: graph node + report file + vector chunks
uv run python -m cli.kb_eval               # vector vs hybrid over the gold question set
uv run python -m cli.memory demo           # working + episodic + semantic memory round-trip
uv run python -m cli.speech_check          # offline Piper TTS → Whisper STT round-trip
uv run python -m cli.text_chat             # inbound query by text (PIN-gated, KB-grounded)
uv run python -m cli.voice                 # offline typed-text voice demo (no audio hardware)
```
