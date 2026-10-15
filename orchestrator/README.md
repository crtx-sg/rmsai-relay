# orchestrator

A hand-rolled turn pipeline (LangGraph-ready: the same nodes can be wrapped later): guardrails → route
(graph template / LLM router / hybrid RAG) → de-id (regex or presidio) → LLM → guardrails → persist.
Plus event persistence, report archival and dispatch. Phases 4 and 7.

- `orchestrator.py`: the turn loop. `chat.py`: builds the orchestrator for text chat (`cli.text_chat`). `guardrails.py`: input/output guardrails (Phase 8).
- `event_flow.py`: persist a `DeviceEvent` to the graph and archive its report narrative to the vector store.
- `bus_consumer.py`: the Redis Stream consumer; criticality gate, then dispatch per the hospital's `outbound.dispatch_mode` (`DISPATCH_MODE` overrides; app worklist push and/or call); an event already acknowledged/resolved (a replay) is persisted but neither pushed nor called (`already_<status>`).
- `outbound_flow.py`: `should_call`, `run_outbound` (retries, then the SMS `fallback_notifier` when unanswered), `run_text_notify`.
- `call_queue.py`: the Redis call queue (`rmsai:calls:queue` sorted set + `rmsai:calls:jobs` hash) and its `CallDispatcher`: most severe first, one call at a time across all consumers (single call slot `rmsai:calls:active`, 60 s lease renewed during the call), skips events acknowledged/resolved while waiting, survives a consumer restart. A call the clinician already answered (right PIN, `rmsai:calls:answered:<event_id>`, 24 h) is not placed again after a crash. Demo limits: a call cut off between the PIN and the end of the alert is not re-placed (the event stays pending on the worklist), and a restarted consumer can re-queue a call that is still ringing.
- `explain.py`: `explain_event`, why an event is (or isn't) shown: rhythm vs threshold, every vitals trigger, criticality escalation, gate decision, one headline.
- `event_log.py`: the `[event]` log line, source labels, `PerfTracker` (rolling summary). `perf_view.py`: the performance view shared by `cli.model_perf` and `POST /metrics`, plus `build_events_view` for `POST /events`.
- `patient_bootstrap.py`, `report.py`.
