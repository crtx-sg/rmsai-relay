"""Episodic memory — past interactions stored as searchable episodes in Qdrant.

Each episode (a clinician question, an answer, an acknowledged alert) is embedded and stored with
session/patient metadata, then recalled by semantic similarity. This is the Mem0-over-Qdrant tier
of the spec; Mem0 can replace the store behind the same `add`/`recall` interface (it adds
LLM-based fact extraction/consolidation on top).

Uses the shared embedder (deterministic hashing by default, BGE when available) so it runs offline.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from common.config import DEFAULT, Config
from kb.vector.embeddings import Embedder, get_embedder

_COLLECTION = "rmsai_episodic"


@dataclass
class Episode:
    text: str
    score: float
    session_id: str | None
    patient_ref: str | None
    timestamp: float


def _point_id(session_id: str | None, text: str, ts: float) -> int:
    h = hashlib.sha256(f"{session_id}|{text}|{ts}".encode()).hexdigest()
    return int(h[:15], 16)  # stable 60-bit id


class EpisodicMemory:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = _COLLECTION) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self._ensure()

    @classmethod
    def from_config(
        cls, config: Config = DEFAULT, embedder_name: str = "auto", collection: str = _COLLECTION
    ) -> "EpisodicMemory":
        return cls(QdrantClient(url=config.qdrant_url), get_embedder(embedder_name), collection)

    @classmethod
    def in_memory(cls, embedder_name: str = "hashing", collection: str = _COLLECTION) -> "EpisodicMemory":
        return cls(QdrantClient(location=":memory:"), get_embedder(embedder_name), collection)

    def _ensure(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )

    def add(
        self,
        text: str,
        *,
        session_id: str | None = None,
        patient_ref: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        ts = timestamp if timestamp is not None else time.time()
        vec = self.embedder.embed([text])[0]
        pid = _point_id(session_id, text, ts)
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "text": text, "session_id": session_id,
                        "patient_ref": patient_ref, "timestamp": ts,
                    },
                )
            ],
        )
        return pid

    def recall(
        self, query: str, k: int = 5, *, patient_ref: str | None = None
    ) -> list[Episode]:
        qfilter = None
        if patient_ref is not None:
            qfilter = Filter(
                must=[FieldCondition(key="patient_ref", match=MatchValue(value=patient_ref))]
            )
        hits = self.client.search(
            self.collection, query_vector=self.embedder.embed([query])[0],
            limit=k, query_filter=qfilter,
        )
        return [
            Episode(
                text=h.payload["text"], score=float(h.score),
                session_id=h.payload.get("session_id"), patient_ref=h.payload.get("patient_ref"),
                timestamp=h.payload.get("timestamp", 0.0),
            )
            for h in hits
        ]
