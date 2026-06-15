"""Semantic memory — long-term knowledge, i.e. the Phase 2A vector index.

A thin adapter over `VectorRetriever` so the three memory tiers expose a uniform recall surface.
Working memory holds the live session, episodic holds past interactions, and semantic holds the
durable clinical corpus (docs + protocol narratives).
"""

from __future__ import annotations

from pathlib import Path

from common.config import DEFAULT, Config
from common.schemas import Passage
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore


class SemanticMemory:
    def __init__(self, retriever: VectorRetriever) -> None:
        self.retriever = retriever

    @classmethod
    def from_config(cls, config: Config = DEFAULT, embedder_name: str = "auto") -> "SemanticMemory":
        store = QdrantStore.connect(config.qdrant_url, collection="rmsai_semantic")
        return cls(VectorRetriever.build(store=store, embedder_name=embedder_name))

    @classmethod
    def in_memory(cls, embedder_name: str = "hashing") -> "SemanticMemory":
        store = QdrantStore.in_memory(collection="rmsai_semantic")
        return cls(VectorRetriever.build(store=store, embedder_name=embedder_name))

    def index_dir(self, directory: str | Path) -> int:
        return self.retriever.index_dir(directory)

    def recall(self, query: str, k: int = 5) -> list[Passage]:
        return self.retriever.retrieve(query, k=k).passages
