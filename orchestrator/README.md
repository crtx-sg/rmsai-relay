# orchestrator

A hand-rolled turn pipeline (LangGraph-ready: the same nodes can be wrapped later): guardrails → route
(graph template / LLM router / hybrid RAG) → de-id (regex or presidio) → LLM → guardrails → persist.
Plus event persistence, report archival and dispatch. Phases 4 and 7.

- `orchestrator.py`: the turn loop. `chat.py`: builds the orchestrator for text chat (`cli.text_chat`). `guardrails.py`: input/output guardrails (Phase 8).
- `event_flow.py`: persist a `DeviceEvent` to the graph and archive its report narrative to the vector store.
- `bus_consumer.py`: the Redis Stream consumer; criticality gate, then dispatch per `DISPATCH_MODE` (app worklist push and/or call).
- `outbound_flow.py`: `should_call`, `run_outbound` (retries, then the SMS `fallback_notifier` when unanswered), `run_text_notify`.
- `explain.py`: `explain_event`, why an event is (or isn't) shown: rhythm vs threshold, every vitals trigger, criticality escalation, gate decision, one headline.
- `event_log.py`: the `[event]` log line, source labels, `PerfTracker` (rolling summary). `perf_view.py`: the performance view shared by `cli.model_perf` and `POST /metrics`.
- `patient_bootstrap.py`, `report.py`.
