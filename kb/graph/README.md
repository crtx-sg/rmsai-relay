# kb/graph

Neo4j schema/migration, patient+document ingestion onto shared nodes, parameterized read-only Cypher templates. Phase 2B.

Also: `events.py` (MonitoredEvent persistence + `list_events`, the event-history query behind `cli.events`, `POST /events` and voice/chat listings), `lookup.py` (question → template), `llm_router.py` (`KB_LLM_ROUTER` fallback), `spoken.py` (STT query normalization), `protocols.py` (care-protocol nodes).
