# kb/eval

Harness: hybrid vs vector-only over the gold set. Phase 2D. Seeding wipes the Neo4j graph
(`reset_all`), so `cli.kb_eval` refuses unless `RMSAI_ALLOW_LIVE_GRAPH_RESET=1`; or pass `--no-seed`.
