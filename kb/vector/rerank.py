"""Reranking of retrieved hits.

* `LexicalReranker` — default, dependency-light: blends the vector score with query-term overlap
  to sharpen ordering. A reasonable baseline stand-in for a cross-encoder.
* `CrossEncoderReranker` — the real BGE reranker (sentence-transformers CrossEncoder), lazy.

A reranker takes the query + hits and returns hits re-sorted (best first), each hit's `score`
replaced by the rerank score.
"""

from __future__ import annotations

import re

from .store import SearchHit

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class LexicalReranker:
    """Blend vector score with query/passage token overlap."""

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        q = _tokens(query)
        if not q:
            return hits
        rescored = []
        for h in hits:
            overlap = len(q & _tokens(h.text)) / len(q)
            score = (1 - self.weight) * h.score + self.weight * overlap
            rescored.append(SearchHit(text=h.text, source=h.source, doc_id=h.doc_id, score=score))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored


class CrossEncoderReranker:
    """Real BGE cross-encoder reranker (sentence-transformers). Lazy import."""

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.text) for h in hits])
        rescored = [
            SearchHit(text=h.text, source=h.source, doc_id=h.doc_id, score=float(s))
            for h, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored
