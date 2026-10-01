# tests

pytest suites, written test-first per phase. The root `conftest.py` sets `RMSAI_NO_DOTENV=1`, so
tests ignore your `.env` and are hermetic.

```bash
uv run pytest -q -m "not infra"   # offline (no containers)
uv run pytest -q                  # full: needs redis/neo4j/qdrant
```

- Markers: `infra` (live stores), `ecgtranscnn` (vendored model).
- **Warning:** the graph/orchestrator infra fixtures reset the **live** Neo4j, so re-ingest afterwards. `-m "not infra"` is safe: `conftest.py` auto-marks the tests using the live-graph fixtures in `test_graph_templates` / `test_hybrid` / `test_voice_orchestrator` as `infra`.
- **Known failures:** 9 pre-existing (`test_episodic_gate` ×3, `test_graph_templates` ×3, `test_speech_check` ×1, `test_voice_config` ×2).
- Suites for recent work:
  - real ECG: `test_real_samples`, `test_ingest_cli`, `test_hdf5_reader`;
  - telephony: `test_telephony_config`, `test_telephony_routing`, `test_phone_worker`, `test_sip_setup`, `test_sip_call`;
  - SMS: `test_sms_fallback`, `test_notify`.
  - model performance & traceability: `test_metrics`, `test_explain`, `test_traceability`, `test_event_log`, `test_gateway_perf`, `test_app_render` (Node, stub DOM; skipped without `node`).
