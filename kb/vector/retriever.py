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

    def index_dir(self, directory: str | Path, *, reset: bool = True) -> int:
        """Chunk, embed, and index every document under `directory`. Returns #chunks.

        `reset=True` (default) recreates the collection first — a clean corpus rebuild. `reset=False`
        appends: chunks are upserted by content-addressed id (idempotent for unchanged docs), so
        re-indexing the corpus preserves other documents already in the store (e.g. event-report
        narratives added by `consume`). The CLI exposes this as `index` (append) vs `index --reset`.
        """
        chunks = chunk_dir(directory)
        if not chunks:
            return 0
        if reset:
            self.store.reset(self.embedder.dim)
        else:
            existing = self.store.vector_dim()
            if existing is not None and existing != self.embedder.dim:
                raise ValueError(
                    f"collection '{self.store.collection}' has vector dim {existing}, but embedder "
                    f"'{self.embedder.name}' produces dim {self.embedder.dim}. Append needs a "
                    f"matching embedder — re-run with that embedder, or rebuild with --reset."
                )
            self.store.ensure(self.embedder.dim)
        vectors = self.embedder.embed([c.text for c in chunks])
        return self.store.index(chunks, vectors)

    def add_document(self, doc_id: str, text: str) -> int:
        """Incrementally chunk + embed + index one document (no reset). Returns #chunks added."""
        from .chunking import chunk_document  # noqa: PLC0415

        chunks = chunk_document(text, doc_id)
        if not chunks:
            return 0
        self.store.ensure(self.embedder.dim)
        vectors = self.embedder.embed([c.text for c in chunks])
        return self.store.index(chunks, vectors)

    def index_paths(self, paths, *, reset: bool = False) -> int:
        """Index a list of documents of **any supported type** (markdown, text, PDF). Returns #chunks.

        `index_dir` only sees `*.md`, which is right for the committed corpus but wrong for uploads —
        a folder of clinical SOPs is mostly PDF. This routes each file through the loader (which
        extracts PDF text and rebuilds its page structure) so an upload survives a corpus rebuild
        instead of being silently skipped by the glob.

        Unreadable files are skipped rather than aborting the batch: one bad scan in an upload folder
        must not stop a rebuild, or the corpus is left half-populated.
        """
        from .chunking import chunk_document  # noqa: PLC0415
        from .loader import read_document  # noqa: PLC0415

        chunks = []
        for path in paths:
            try:
                chunks.extend(chunk_document(read_document(path), Path(path).name))
            except Exception as exc:  # noqa: BLE001 - reported by the caller's log, not fatal
                print(f"[index] skipping {path}: {type(exc).__name__}: {exc}", flush=True)
        if not chunks:
            return 0
        if reset:
            self.store.reset(self.embedder.dim)
        else:
            self.store.ensure(self.embedder.dim)
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
