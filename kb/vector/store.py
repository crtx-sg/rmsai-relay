"""Qdrant-backed vector store.

Thin wrapper over `qdrant-client`. Works against a live server (`QdrantStore.connect(url)`) or an
in-memory instance (`QdrantStore.in_memory()`, used by tests — no server needed). Stores chunk
text + citation in the point payload so search returns ready-to-cite passages.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk

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

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunks + their vectors. Returns the number indexed."""
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"text": c.text, "source": c.source, "doc_id": c.doc_id},
            )
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

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
