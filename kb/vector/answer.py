"""Grounded, cited answering over retrieved passages — with an out-of-corpus decline.

`answer(query, retriever, llm=None)` retrieves passages and:
  * declines (no fabrication) when nothing relevant is retrieved;
  * otherwise produces a grounded answer with citations. With an `LLMProvider` it asks the model
    to answer **only** from the supplied context; without one it falls back to an extractive
    answer (the most relevant passage), so the path runs with no LLM for tests/offline.

Relevance is gated on lexical overlap between the query's content terms and the best passage,
which is embedder-agnostic and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.interfaces import LLMProvider

from .retriever import VectorRetriever

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "what", "how", "do", "does", "i", "should", "with", "at", "be", "my", "we", "you", "it",
    "this", "that", "if", "when", "which", "can", "will", "from", "by", "as",
}
_DECLINE = "I don't have information on that in the knowledge base."
_DEFAULT_MIN_OVERLAP = 0.18


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    declined: bool = False
    passages: list = field(default_factory=list)  # list[Passage]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _best_overlap(query: str, passages) -> float:
    q = _content_tokens(query)
    if not q:
        return 0.0
    return max((len(q & _content_tokens(p.text)) / len(q) for p in passages), default=0.0)


def _build_prompt(query: str, passages) -> str:
    context = "\n\n".join(f"[{i + 1}] (source: {p.source})\n{p.text}" for i, p in enumerate(passages))
    return (
        "Answer the question using ONLY the context below. If the context does not contain the "
        "answer, say you don't know. Cite sources as [n].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def answer(
    query: str,
    retriever: VectorRetriever,
    llm: LLMProvider | None = None,
    *,
    k: int = 4,
    min_overlap: float = _DEFAULT_MIN_OVERLAP,
) -> GroundedAnswer:
    result = retriever.retrieve(query, k=k)
    passages = result.passages

    if not passages or _best_overlap(query, passages) < min_overlap:
        return GroundedAnswer(answer=_DECLINE, declined=True, passages=passages)

    citations = list(dict.fromkeys(p.source for p in passages))  # unique, order-preserving
    if llm is not None:
        text = llm.generate(_build_prompt(query, passages))
    else:
        # Extractive fallback: the single most relevant passage is the grounded answer.
        text = passages[0].text
    return GroundedAnswer(answer=text, citations=citations, declined=False, passages=passages)
