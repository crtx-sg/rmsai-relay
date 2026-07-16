# rmsai-relay — POC → Production plan

High-level plan to take the POC to a robust, scalable, multi-tenant production platform.
Estimates are **order-of-magnitude for planning**, assuming the POC has de-risked the core loop
(it has). See [../ARCHITECTURE.md](../ARCHITECTURE.md) for how the system works today.

## Where the POC is

Proves the full loop end-to-end on **one node** (Docker Compose), **one hospital**, **synthetic
data**: HDF5/MQTT → ECG classification → graph + vector KB → criticality-gated outbound call →
grounded voice/text Q&A. Single-instance Redis stream, Neo4j, Qdrant; self-hosted LiveKit + Ollama.
**Not yet:** multi-tenancy, HA, RBAC, observability, compliance, or **continuous streaming telemetry**
(live MQTT→WebRTC waveforms/vitals were the deferred POC Phase 9).

## Target

Multi-tenant, HIPAA-grade, self-healing platform — **100s of hospitals**, high physiological-event
and call throughput, observable, extensible.

## Workstreams (gaps to close)

| # | Workstream | Key work |
|---|---|---|
| 1 | **Multi-tenancy & isolation** | per-hospital tenancy, KB namespacing (Qdrant collections / Neo4j subgraphs), PHI segregation, tenant routing |
| 2 | **Streaming telemetry** | continuous device streams: **edge MQTT gateways → partitioned stream bus**, live waveform/vitals **MQTT→WebRTC** to the app, edge store-and-forward, sliding-window (not just HDF5) inference |
| 3 | **Event bus & processing** | Kafka (or Redis Streams consumer groups) **partitioned by hospital+patient**, dead-letter queues, idempotency, backpressure |
| 4 | **Model serving** | ECG on Triton/GPU (batching, autoscale), LLM on a vLLM cluster, embeddings service; model registry + versioning |
| 5 | **Voice at scale** | LiveKit cluster/Cloud, SIP trunk capacity, **K8s worker autoscaling** (job-per-room), concurrency limits |
| 6 | **Data stores HA** | Neo4j cluster, Qdrant sharding, **time-series DB** (Timescale/Influx) for waveforms/vitals, managed Kafka/Redis/Postgres, object store |
| 7 | **Observability** | OpenTelemetry tracing, Prometheus/Grafana, SLOs, **PHI-safe** structured logs, audit pipeline, model-drift + call-quality + stream-lag metrics |
| 8 | **Reliability** | retries / circuit-breakers, graceful degradation, DR + backup, chaos + load testing, rate limiting |
| 9 | **Security & compliance** | HIPAA / SOC2, vendor BAAs, KMS encryption at rest + in transit, RBAC/SSO, de-id validation, pen-test |
| 10 | **Clinical safety / regulatory** ⚠️ | guardrail eval harness, red-team, clinical review — **SaMD / FDA-510(k) path if this is decision-support** |
| 11 | **Integration** | FHIR/HL7 EHR at scale, per-site MQTT device gateways, **edge deploy** for on-prem PHI |
| 12 | **MLOps + DevEx** | retraining / eval / benchmark, drift detection, A/B; K8s + Terraform IaC, CI/CD, multi-region, feature flags |

## Phased plan (~12–15 months to GA; phases overlap)

| Phase | Months | Focus |
|---|---|---|
| 0 · Foundations | 0–2 | Architecture, IaC/K8s, multi-tenant model, security baseline, CI/CD |
| 1 · Scalable core | 2–6 | Partitioned bus, streaming telemetry (edge MQTT → stream + time-series), model serving, HA data stores, tenancy |
| 2 · Voice + clinical loop | 4–8 | Telephony scale, worker autoscale, live waveform (MQTT→WebRTC) to app, orchestrator / guardrail hardening |
| 3 · Obs + reliability + compliance | 5–10 | Observability stack, DR, HIPAA/SOC2 audit |
| 4 · MLOps + EHR integration | 7–12 | Retrain/eval, FHIR/HL7, edge deployment |
| 5 · Pilot → GA | 10–15 | 2–3 hospital pilot, load/chaos, staged rollout |

## Team (~18–24, cross-functional)

Platform/Infra/SRE **5** · Backend services **5** · ML/MLOps **3** · Voice/telephony **2** ·
Data eng **2** · Security/Compliance **2** · Frontend **2** · QA/automation **2** ·
EM/PM/Architect **3** · Clinical/Regulatory SME **1–2**.

## AI-tooling headcount optimization

With Claude Code + agentic dev tooling, savings track how much of an area is **code you write**
(scaffolding, IaC, tests, glue, refactors — high AI leverage) vs. **judgment / coordination /
domain expertise** (incident response, model & eval design, regulatory sign-off, coordination — low
leverage). Planning-grade; the saved FTE isn't free — AI-generated code raises the senior-review load.

| Area | Baseline | AI leverage | Optimize | Net | Why |
|---|---|---|---|---|---|
| Platform/Infra/SRE | 5 | Med-High | ~1 | ~4 | IaC/K8s/CI-CD templatable; on-call + capacity judgment isn't |
| Backend services | 5 | **High** | ~1.5–2 | ~3–3.5 | Service scaffolding, APIs, tests — the sweet spot |
| ML/MLOps | 3 | Low-Med | ~0.5 | ~2.5 | Serving/pipeline config yes; model + eval design is novel work |
| Voice/telephony | 2 | Med | ~0.5 | ~1.5 | LiveKit/SIP glue helped; specialized, already small |
| Data eng | 2 | High | ~0.5 | ~1.5 | ETL, schema, pipeline code |
| Security/Compliance | 2 | Low | ~0.5 | ~1.5 | AI drafts policy/evidence/docs; audit + sign-off is human |
| Frontend | 2 | **High** | ~0.5–1 | ~1–1.5 | Component code, forms, state |
| QA/automation | 2 | **High** | ~1 | ~1 | Test generation is a prime AI use case |
| EM/PM/Architect | 3 | Low | ~0.5 | ~2.5 | Specs/docs helped; coordination is the job |
| Clinical/Regulatory SME | 1–2 | ~None | 0 | 1–2 | Pure domain expertise; no code leverage |
| **Total** | **~27** | — | **~7–8 FTE** | **~19–20** | Lands at the low end of the ~18–24 headline |

Biggest wins: Backend, QA, Frontend, Data eng, Infra. ~Zero in Clinical/Regulatory and little in
EM/PM and Security-audit — the same long poles that dominate this product's risk.

## Rough monthly run-rate at 100s-hospital scale

Cloud, order-of-magnitude, load-dependent.

| Item | ~$/mo |
|---|---|
| GPU compute — LLM (vLLM) + Whisper STT + ECG inference, autoscaled | $40–90k |
| Managed data — Neo4j, Qdrant, Kafka, Redis, Postgres, **time-series**, object store | $15–35k |
| Streaming / edge — ingress bandwidth, edge gateways, WebRTC egress | $5–15k |
| Telephony / LiveKit (per-minute + infra) | $5–20k |
| Observability (Datadog / Grafana Cloud) | $5–15k |
| **Infra subtotal** | **~$70–175k/mo** |
| Dev AI tooling (Claude Code, ~20 seats — Max-plan seats to cap the metered tail) | ~$2.5–6k/mo (**~$30–75k/yr**) |
| One-time: HIPAA/SOC2 audit + pen-test | $50–150k |

**Dev-tooling NRE (per annum): ~$30–75k/yr** (~$50k mid-case, ~20 coding seats). Prefer Max-plan
seats over raw metered API — heavy agentic runs can otherwise spike one dev to $500–1k/mo. Against
the ~7–8 FTE it offsets (~$1.4–1.6M/yr loaded), that's a **~30× return** and a rounding error next
to a single line of the infra run-rate.

## Long poles / risks

- ⚠️ **Regulatory (SaMD)** classification can shift the timeline materially — clarify first.
- Clinical validation of the ECG model + LLM guardrails.
- PHI compliance sign-off.
- **PHI constraint = self-hosted STT/LLM only** (no cloud AI on real data) — this drives the GPU cost.

**De-risk with a 2–3 hospital pilot before GA.**
