# tests

pytest suites, written test-first per phase. The root `conftest.py` sets `RMSAI_NO_DOTENV=1`, so
tests ignore your `.env` and are hermetic.

```bash
uv run pytest -q -m "not infra"   # offline (no containers)
uv run pytest -q                  # full: needs redis/neo4j/qdrant
```

- Markers: `infra` (live stores), `ecgtranscnn` (vendored model).
- **Live graph guard:** any test that would `reset_all()` the live Neo4j is skipped unless `RMSAI_ALLOW_LIVE_GRAPH_RESET=1` (`conftest.py`), however it is selected (file, `-k`, or no `-m`). The live-graph-fixture tests in `test_graph_templates` / `test_hybrid` / `test_voice_orchestrator` are also auto-marked `infra`. With the opt-in set the run wipes the graph, so re-ingest afterwards.
- **Known failures:** 3 pre-existing in the offline run (`test_speech_check` ×1, `test_voice_config` ×2; checked 2026-10-13). The full suite has also shown `test_episodic_gate` ×3.
- Suites for recent work:
  - real ECG: `test_real_samples`, `test_ingest_cli`, `test_hdf5_reader`;
  - telephony: `test_telephony_config`, `test_telephony_routing`, `test_phone_worker`, `test_sip_setup`, `test_sip_call`;
  - SMS: `test_sms_fallback`, `test_notify`.
  - call queue & event history: `test_call_queue`, `test_event_history`, `test_verifiable_listings`, `test_gateway_events`, `test_voice_on_select`.
  - model performance & traceability: `test_metrics`, `test_explain`, `test_traceability`, `test_event_log`, `test_gateway_perf`, `test_app_render` (Node, stub DOM; skipped without `node`).
