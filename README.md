# rmsai-relay

Self-hostable POC: medical IoT + AI. Ingests physiological event data (HDF5 archives / MQTT
stream), classifies clinically significant events with the `ECG_TransConv` model, reports them to a
remote clinician over the phone, and answers the clinician's follow-up questions in voice or text —
grounded in a clinical knowledge base and a per-patient knowledge graph.

Built **leaf-up, test-first, one phase at a time**. See `CLAUDE.md` for the working agreement.

## Quick start (Phase 0)

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

## Layout

`common/` contracts · `ingest/` readers · `inference/` model+analysis · `kb/` vector+graph+hybrid ·
`memory/` tiers · `orchestrator/` LangGraph · `voice/` SIP+LiveKit · `app/`+`live/` companion app ·
`emr/` FHIR · `cli/` entrypoints · `infra/` compose · `external/ecgtranscnn/` vendored ·
`data/{synthetic,fixtures}`.
