# docs

Small committed clinical-protocol corpus for the knowledge base. Phase 2A indexes these into the
vector store (Qdrant); Phase 2B also extracts `Condition`/`Treatment`/`Guideline` entities from the
same text into the graph, onto shared nodes. Kept deliberately small and deterministic so retrieval
and evaluation are reproducible.
