"""Qdrant-backed vector store.

Thin wrapper over `qdrant-client`. Works against a live server (`QdrantStore.connect(url)`) or an
in-memory instance (`QdrantStore.in_memory()`, used by tests — no server needed). Stores chunk
text + citation in the point payload so search returns ready-to-cite passages.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk


def _stable_id(chunk: Chunk) -> int:
    """Content-addressed point id so re-indexing the same chunk is idempotent (upsert)."""
    h = hashlib.sha256(f"{chunk.doc_id}|{chunk.idx}|{chunk.source}".encode()).hexdigest()
    return int(h[:15], 16)

DEFAULT_COLLECTION = "rmsai_docs"


@dataclass
class SearchHit:
    text: str
    source: str
    doc_id: str
    score: float


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str = DEFAULT_COLLECTION) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def in_memory(cls, collection: str = DEFAULT_COLLECTION) -> "QdrantStore":
        return cls(QdrantClient(location=":memory:"), collection)

    @classmethod
    def connect(cls, url: str, collection: str = DEFAULT_COLLECTION) -> "QdrantStore":
        return cls(QdrantClient(url=url), collection)

    def reset(self, dim: int) -> None:
        """(Re)create the collection with the given vector dimension (idempotent)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def ensure(self, dim: int) -> None:
        """Create the collection if it does not exist (for incremental adds)."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def vector_dim(self) -> int | None:
        """Return the collection's configured vector size, or None if it does not exist."""
        if not self.client.collection_exists(self.collection):
            return None
        params = self.client.get_collection(self.collection).config.params.vectors
        # Single unnamed vector -> VectorParams(.size); named vectors -> dict (take the first).
        if isinstance(params, dict):
            params = next(iter(params.values()))
        return params.size

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunks + their vectors (content-addressed ids → idempotent). Returns the count."""
        points = [
            PointStruct(
                id=_stable_id(c),
                vector=vec,
                payload={"text": c.text, "source": c.source, "doc_id": c.doc_id},
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def chunks_for_doc(self, doc_id: str, limit: int = 100) -> list[dict]:
        """Return the payloads ({text, source, doc_id}) of every chunk for a document id.

        Inspection/dump helper — filters by payload `doc_id` (no vector search). Empty if the
        collection or document is absent.
        """
        from qdrant_client.models import (  # noqa: PLC0415
            FieldCondition,
            Filter,
            MatchValue,
        )

        if not self.client.collection_exists(self.collection):
            return []
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        points, _ = self.client.scroll(
            collection_name=self.collection, scroll_filter=flt, limit=limit,
            with_payload=True, with_vectors=False,
        )
        return [p.payload for p in points]

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchHit]:
        hits = self.client.search(
            collection_name=self.collection, query_vector=query_vector, limit=k
        )
        return [
            SearchHit(
                text=h.payload["text"],
                source=h.payload["source"],
                doc_id=h.payload["doc_id"],
                score=float(h.score),
            )
            for h in hits
        ]
