"""Grounded answering over the two labelled blocks (passages + relationships).

Keeps the split visible to the LLM: the prompt presents *Retrieved passages* and *Known
relationships* as distinct, separately-cited sections (no cross-block merge). Without an LLM an
extractive answer is assembled from whichever block is relevant, so the path runs offline.
Citations trace back to the contributing block.
"""

from __future__ import annotations

from common.interfaces import LLMProvider
from kb.vector.answer import _DECLINE, GroundedAnswer, _best_overlap

from .retriever import HybridRetriever

_MIN_OVERLAP = 0.18


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
) -> GroundedAnswer:
    result = retriever.retrieve(query, mode=mode, k=k)
    passages, relationships = result.passages, result.relationships

    passage_relevant = bool(passages) and _best_overlap(query, passages) >= min_overlap
    if not relationships and not passage_relevant:
        return GroundedAnswer(answer=_DECLINE, declined=True, passages=passages)

    citations: list[str] = []
    if relationships:
        citations += [r.source for r in relationships]
    if passage_relevant:
        citations += [p.source for p in passages]
    citations = list(dict.fromkeys(citations))  # unique, order-preserving

    if llm is not None:
        text = llm.generate(_build_prompt(query, passages, relationships))
    else:
        # Extractive: lead with graph relationships (the graph-only facts), then the top passage.
        parts: list[str] = []
        if relationships:
            parts.append(
                "Known relationships:\n" + "\n".join(f"- {r.fact}" for r in relationships)
            )
        if passage_relevant:
            parts.append(passages[0].text)
        text = "\n\n".join(parts)

    return GroundedAnswer(answer=text, citations=citations, declined=False, passages=passages)
