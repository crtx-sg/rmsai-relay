# rmsai-relay

Self-hostable POC for **medical IoT + AI**. It ingests physiological event data (HDF5 archives / an
MQTT stream), classifies clinically-significant arrhythmias with the `ECG_TransConv` model, persists
each event into a graph + vector knowledge base, **alerts a remote clinician** about critical events
(a companion-app worklist, a voice call over WebRTC or the phone, or SMS), and then answers the
clinician's follow-up questions — by voice or text — grounded in a clinical knowledge base and a
per-patient knowledge graph.

Built **leaf-up, test-first, one phase at a time**. Every subsystem ships a `pytest` suite and a
runnable CLI harness before it is wired upward. See [`CLAUDE.md`](CLAUDE.md) for the full working
agreement; the source-of-truth design lives in the project spec + kickoff prompt.

**Running a demo?** [`DEMO.md`](DEMO.md) is the step-by-step runbook: real ECG → real_v2 →
worklist/voice → grounded Q&A, with a troubleshooting table.

## Documentation

| File | Read it when you want to |
|---|---|
| [`INSTALL.md`](INSTALL.md) | install and bring the stack up (Docker or host), make targets, optional extras |
| [`CONFIGURATION.md`](CONFIGURATION.md) | change a setting: stubs, the LLM, the call decision, per-hospital files, vital trends |
| [`OPERATIONS.md`](OPERATIONS.md) | run it day to day: initialize the KB, drive events, inspect, measure, shut down; query reference; model performance |
| [`TESTING.md`](TESTING.md) | verify it works: speech engines, the knowledge base, end-to-end paths, per-subsystem CLIs |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | understand how it works: pipelines, voice loop, contracts, tech stack, KB data model |
| [`DEMO.md`](DEMO.md) | run the real-ECG demo, including local-only mode |
| [`TELEPHONY_SETUP.md`](TELEPHONY_SETUP.md) | ring a real phone: LiveKit Cloud + SignalWire, call flow, troubleshooting |
| [`CLAUDE.md`](CLAUDE.md) | the working agreement (hard rules) |

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
4. **Report** critical events to the companion-app worklist (all at once) and/or by an outbound
   voice call: WebRTC, or a real phone call via LiveKit Cloud + SignalWire
   ([Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire)). Phone calls queue in Redis and
   go out one at a time, most severe first. A text message is the alternative channel, and the SMS
   fallback covers a call that goes unanswered (including a pickup where no PIN is entered; a PIN
   lockout is not texted).
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
  surface, and a companion app (worklist, chat, ack, scoped artifacts; live waveform streaming is
  still planned). Voice & chat are the
  **control plane**; the companion app is the **data plane** for waveforms/video voice can't carry.

### Design principles (hard rules)

- **Self-hosted by default; provider abstraction always.** All real/PHI processing runs against
  local models. The LLM sits behind one `LLMProvider` interface: `EchoLLM` (offline, deterministic)
  is the code default, `LLM_PROVIDER=ollama` selects the self-hosted model, and `anthropic`,
  `gemini` or `openai` (any OpenAI-compatible API) select a cloud model behind the same interface
  (see [Choosing the LLM](CONFIGURATION.md#choosing-the-llm)). Cloud APIs are used **only** on synthetic data, never
  PHI.
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

![rmsai-relay architecture and pipeline flow](architecture.png)

*Regenerate the diagram with `uv run python docs/make_arch_diagram.py`.*

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
                                │  "rmsai.events"    │   (consumer group; ack when handled,
                                └─────────┬─────────┘    else retry → rmsai.events.dead; MAXLEN ~100k)
                       cli.consume        │  XREADGROUP
                       (dict_to_event)    ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                            ORCHESTRATE  (orchestrator/)                         │
        │  ensure_patient (G8 auto-create: bed assign + synthetic history → graph)        │
        │  process_device_event   ── persist ──►  Neo4j MonitoredEvent + ActionItems      │
        │                          ── archive ──►  Qdrant report narrative + Report node   │
        │  event_criticality(event, config) → should_call gate                            │
        │        │ below threshold / NORMAL_SINUS → skip (still persisted)                │
        │        │ already acknowledged/resolved (a replay) → not re-alerted              │
        │        ▼ critical                                                               │
        │  ① app:  push the worklist row to the inbox room NOW (every gated event)        │
        │  ② call: SIP → queue it in Redis (ZSET rmsai:calls:queue, most severe first)    │
        │          WebRTC / text (SMS) → dispatch inline                                  │
        │  CallDispatcher: claim the call slot → ZPOPMIN → skip if acked → one call       │
        └───────────────────────────────┬────────────────────────────────────────────────┘
                                         │  stage OutboundAlert at dial time (Redis, keyed by room)
                                         ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                         VOICE LOOP  (two-process model)                         │
        │  relay PLACES the call (LiveKitCaller → SIP dial / WebRTC room)                  │
        │  cli.voice_worker JOINS the room (Handler is the 'LLM' node; stub LLM fills the  │
        │  pipeline gate so llm_node runs):                                                │
        │   audio ─Whisper STT─►[wake word]─► OutboundHandler ─LLM(RAG)────────► Piper TTS │
        │   text  ─chat box────────────────► OutboundHandler ─LLM(RAG)────────► chat text  │
        │     PIN gate → speak THIS event's alert → Q&A grounded in KB+graph → "ack"       │
        │     → MonitoredEvent.status = acknowledged                                       │
        │  modality-matched: audio→audio (wake-gated), text→text; never crossed            │
        └──────────────────────────────────────────────────────────────────────────────┘

  Inbound path (clinician calls/joins to query): same worker, build_handler() → PIN gate →
  intent → operational Cypher template OR hybrid retrieve → de-id → LLM → cited answer.
```

---

## Phase status

Built leaf-up, one phase at a time; a phase is "done" only when its CLI test passes. Phases **0–8
are complete** and the companion app (9a) ships; live media (9b) is deferred. Outbound phone calls
over SignalWire work end to end; phone call-in is still being tested (see
[Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire)).

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
| 9a | Companion app — worklist, event-scoped chat/voice, ack, scoped artifact links (`app/`, `live/`, `cli.gateway`) | ✅ |
| 9b | Live media — MQTT→WebRTC ECG/vitals streaming, camera relay, consent/audit | 🔜 planned |
| — | Bus consumer + event-driven LiveKit/WebRTC outbound (`cli.consume`) | ✅ |
| — | Real-ECG model `real_v2` (13-class 5-fold ensemble) + ecg_sigma HDF5 reader | ✅ |
| — | Held-out real-ECG curation (`cli.real_samples`, `PT9#####` pseudonyms) + `cli.ingest --dir` scoring | ✅ |
| T1–T4 | Telephony split: `Config.telephony()`, `cli.sip_setup`, call paths → telephony LiveKit, phone worker (`make phone-up`) | ✅ |
| T5 | SMS fallback on unanswered calls (a pickup with no PIN counts as unanswered; a PIN lockout is not texted): SignalWire (`--notifier signalwire`, same number as the calls; `make demo-phone-up` default); a simulated SMS is recorded as `sms_simulated`, never as delivered | ✅ test SMS delivered to +91 (2026-10-13) |
| C1–C2 | SignalWire carrier: `TELEPHONY_CARRIER`, native SIP trunk via `cli.sip_setup` (`--swml`) | ✅ outbound (2026-10-12) and call-in (2026-10-13) verified live |
| — | Call queue: app gets every event at once, calls one at a time in severity order, skip-if-acked, restart-safe (`orchestrator/call_queue.py`, `make demo-phone-up`) | ✅ |
| — | Live-call hardening (2026-10-28/29): each redial in a fresh room, hang up and ring again after a LiveKit reconnect, keypad PIN, protocol answers verbatim (undocumented rhythm: "not known"), deteriorating-vitals and who-is-on-bed answers | ✅ unit-tested; fresh-room retry and keypad PIN not yet verified live |
| — | Phone worker registered on the wrong LiveKit (agents CLI env override) | ✅ fixed (`fe89028`) |
| E1–E5 | Model performance & traceability: outcome + metrics, "why shown", provenance + model id, `[event]`/`[perf]` logs, `cli.model_perf`, app tab + detail panel | ✅ |
| — | Voice on select: header toggle, muted by default in `app+call` so a clinician on the phone alert doesn't hear the event twice (`/select-silent`) | ✅ |
| — | KB chunks cite their parent heading (`file#Parent › Child`); horizontal rules are not content | ✅ |
| H1–H3 | Event history: one `pending` definition, `list_events` + `cli.events` (`--ask`), verifiable voice/chat listings that state their filter (or decline), app **Events** tab (`POST /events`, merges model performance), worklist backfill on login, chat → Events-tab link | ✅ |
| E6 | Ask "why am I seeing this?" / "how is the model doing?" by chat or voice | 🔜 optional |
| — | File-drop auto-publish watcher (inotify → `cli.ingest --emit bus`) | 🔜 planned |

---

### Safety & PHI guarantees

- No PHI in logs (redacting logger; patients referenced by pseudonym only).
- De-identification asserted on model inputs in tests; **fails closed** (no LLM call on de-id error).
- Caller authentication enforced before any PHI is voiced; the graph is queried only through fixed,
  read-only, parameterized templates (the LLM router can only pick one), and raw Cypher in the
  `cli.graph lookup` dev path is refused unless read-only.
- Append-only **JSONL audit log** from Phase 0: `{ts, actor, action, subject(pseudonym), outcome}`
  for PHI reads, outbound calls + outcome, inbound auth results, and queries.
- Idempotent ingestion (MERGE by event `uuid`) dedupes MQTT+HDF5 and replays.
