"""Vector retriever: ties embedder + Qdrant store (+ optional reranker) together.

`index_dir` chunks + embeds + indexes a corpus; `retrieve` embeds the query, searches, optionally
reranks, and returns a `RetrievalResult` with the **passages block only** (the graph
`relationships` block stays empty under `vector` mode — decision D8).
"""

from __future__ import annotations

from pathlib import Path

from common.schemas import Passage, RetrievalResult

from .chunking import chunk_dir
from .embeddings import Embedder, get_embedder
from .rerank import LexicalReranker
from .store import QdrantStore, SearchHit


class VectorRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, reranker=None) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    @classmethod
    def build(
        cls,
        store: QdrantStore | None = None,
        embedder_name: str = "auto",
        *,
        rerank: bool = True,
    ) -> "VectorRetriever":
        embedder = get_embedder(embedder_name)
        store = store or QdrantStore.in_memory()
        return cls(store, embedder, LexicalReranker() if rerank else None)

    def index_dir(self, directory: str | Path) -> int:
        """Chunk, embed, and index every document under `directory`. Returns #chunks."""
        chunks = chunk_dir(directory)
        if not chunks:
            return 0
        self.store.reset(self.embedder.dim)
        vectors = self.embedder.embed([c.text for c in chunks])
        return self.store.index(chunks, vectors)

    def search(self, query: str, k: int = 5, *, rerank: bool = True) -> list[SearchHit]:
        qvec = self.embedder.embed([query])[0]
        # Over-fetch a little before reranking so the reranker can reorder a wider pool.
        hits = self.store.search(qvec, k=k * 2 if (rerank and self.reranker) else k)
        if rerank and self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]

    def retrieve(self, query: str, k: int = 5, *, rerank: bool = True) -> RetrievalResult:
        hits = self.search(query, k=k, rerank=rerank)
        passages = [Passage(text=h.text, source=h.source, score=h.score) for h in hits]
        return RetrievalResult(query=query, passages=passages, relationships=[], mode="vector")
