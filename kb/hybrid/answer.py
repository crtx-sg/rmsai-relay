"""Grounded answering over the two labelled blocks (passages + relationships).

Keeps the split visible to the LLM: the prompt presents *Retrieved passages* and *Known
relationships* as distinct, separately-cited sections (no cross-block merge). Without an LLM an
extractive answer is assembled from whichever block is relevant, so the path runs offline.
Citations trace back to the contributing block.
"""

from __future__ import annotations

from common.config import DEFAULT
from common.interfaces import LLMProvider
from kb.vector.answer import _DECLINE, GroundedAnswer, _best_overlap

from .retriever import HybridRetriever

_MIN_OVERLAP = 0.18


def is_relevant(result, overlap: float, top_score: float, *,
                min_relevance: float, min_overlap: float = _MIN_OVERLAP) -> bool:
    """Whether retrieval found something worth answering from.

    Three independent ways to vouch, any one of which is enough: the graph returned a fact, the best
    passage *means* the same thing (semantic similarity, which survives rewording), or it *says* the
    same words. The lexical arm alone used to decide — which refused correctly-retrieved passages
    that happened to be phrased differently from the question, the single largest source of
    "I don't have information on that". The two arms fail in different directions, so keeping both
    costs nothing: word overlap carries the hashing embedder, whose similarity scores do not
    separate on-topic from off-topic at all.
    """
    if result.relationships:
        return True
    if not result.passages:
        return False
    return top_score >= min_relevance or overlap >= min_overlap


def _build_prompt(query: str, passages, relationships) -> str:
    pblock = "\n\n".join(
        f"[P{i + 1}] (source: {p.source})\n{p.text}" for i, p in enumerate(passages)
    ) or "(none)"
    rblock = "\n".join(
        f"[R{i + 1}] (source: {r.source}) {r.fact}" for i, r in enumerate(relationships)
    ) or "(none)"
    return (
        "Answer the question using ONLY the context below, which has two parts. Cite passages as "
        "[Pn] and relationships as [Rn]. If neither part answers it, say you don't know.\n\n"
        f"## Retrieved passages\n{pblock}\n\n## Known relationships\n{rblock}\n\n"
        f"Question: {query}\nAnswer:"
    )


def answer(
    query: str,
    retriever: HybridRetriever,
    llm: LLMProvider | None = None,
    *,
    mode: str = "hybrid",
    k: int = 4,
    min_overlap: float = _MIN_OVERLAP,
    min_relevance: float | None = None,
) -> GroundedAnswer:
    """Answer offline (extractive) or with `llm`, behind the SAME relevance gate as the live
    orchestrator (`is_relevant`), so `cli.kb_eval` scores the decline behaviour that ships."""
    result = retriever.retrieve(query, mode=mode, k=k)
    passages, relationships = result.passages, result.relationships
    min_relevance = DEFAULT.kb_min_relevance if min_relevance is None else min_relevance

    overlap = _best_overlap(query, passages) if passages else 0.0
    top_score = passages[0].score if passages else 0.0
    if not is_relevant(result, overlap, top_score, min_relevance=min_relevance,
                       min_overlap=min_overlap):
        return GroundedAnswer(answer=_DECLINE, declined=True, passages=passages)
    # Passages are quoted only when they vouch for themselves (not just because a graph fact did).
    passage_relevant = bool(passages) and (top_score >= min_relevance or overlap >= min_overlap)

    citations: list[str] = []
    if relationships:
        citations += [r.source for r in relationships]
    if passage_relevant:
        citations += [p.source for p in passages]
    citations = list(dict.fromkeys(citations))  # unique, order-preserving

    if llm is not None:
        text = llm.generate(_build_prompt(query, passages, relationships))
    else:
        # Extractive: lead with graph relationships (the graph-only facts), then the retrieved
        # passage context (all top-k, since the answer fact may live in any retrieved chunk —
        # an LLM would synthesize across them).
        parts: list[str] = []
        if relationships:
            parts.append(
                "Known relationships:\n" + "\n".join(f"- {r.fact}" for r in relationships)
            )
        if passage_relevant:
            parts.append("\n\n".join(p.text for p in passages))
        text = "\n\n".join(parts)

    return GroundedAnswer(answer=text, citations=citations, declined=False, passages=passages)
