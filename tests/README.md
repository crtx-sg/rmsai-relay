# tests

pytest suites, written test-first per phase. The root `conftest.py` sets `RMSAI_NO_DOTENV=1`, so
tests ignore your `.env` and are hermetic.

```bash
uv run pytest -q -m "not infra"   # offline (no containers)
uv run pytest -q                  # full: needs redis/neo4j/qdrant
```

- Markers: `infra` (live stores), `ecgtranscnn` (vendored model).
- **Live graph guard:** any test that would `reset_all()` the live Neo4j is skipped unless `RMSAI_ALLOW_LIVE_GRAPH_RESET=1` (`conftest.py`), however it is selected (file, `-k`, or no `-m`). The live-graph-fixture tests in `test_graph_templates` / `test_hybrid` / `test_voice_orchestrator` are also auto-marked `infra`. With the opt-in set the run wipes the graph, so re-ingest afterwards.
- **Known failures:** none in the offline run (1157 passed, 2026-10-14). The full suite (stores up) fails `test_orchestrator::test_multi_turn_state_persists`: with the hashing embedder the question does not pass the relevance gate.
- Suites for recent work:
  - real ECG: `test_real_samples`, `test_ingest_cli`, `test_hdf5_reader`;
  - telephony: `test_telephony_config`, `test_telephony_routing`, `test_phone_worker`, `test_sip_setup`, `test_sip_call`;
  - SMS: `test_sms_fallback`, `test_notify`.
  - call queue & event history: `test_call_queue`, `test_event_history`, `test_verifiable_listings`, `test_gateway_events`, `test_voice_on_select`.
  - model performance & traceability: `test_metrics`, `test_explain`, `test_traceability`, `test_event_log`, `test_gateway_perf`, `test_app_render` (Node, stub DOM; skipped without `node`).
  - reliability: `test_consume_ack` (retry/dead-letter), `test_graph_status` (status set on create, `--reopen`, one current bed, patient ranking), `test_vector_replace`, `test_linker`, `test_config_env`, `test_event_chain`;
  - voice/chat answers: `test_query_coverage`, `test_call_2026_10_14`, `test_review_2026_10_14`, `test_patient_ranking`, `test_patient_scope_answers`.
