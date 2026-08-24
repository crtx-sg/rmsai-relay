"""Why a turn declined — the three causes need opposite fixes, so the log has to tell them apart.

`I don't have information on that in the knowledge base` is the same sentence whether the corpus is
empty, the retriever missed, or the relevance gate rejected a perfectly good passage. These pin the
diagnosis that distinguishes them.
"""

from __future__ import annotations

from common.schemas import Passage, Relationship, RetrievalResult
from orchestrator.orchestrator import explain_decline, is_relevant


def _result(passages=(), relationships=()):
    return RetrievalResult(query="q", passages=list(passages), relationships=list(relationships),
                           mode="hybrid")


def test_graph_hit_is_never_a_decline():
    assert explain_decline(_result(relationships=[Relationship(fact="afib CO_MORBID_WITH htn", source="graph")]), 0.0) == ""


def test_empty_retrieval_points_at_the_corpus():
    detail = explain_decline(_result(), 0.0)
    assert "no vector passages and no graph relationships" in detail
    assert "docs/" in detail and "re-index" in detail  # the actionable fix, not just the symptom


def test_rejected_passage_reports_both_signals_and_the_top_source():
    # The confusing case: retrieval worked and BOTH relevance signals still said no. The log has to
    # show each number against its threshold, or there is no way to tell which one to move.
    detail = explain_decline(_result([Passage(text="…", source="vt_vf.md#Pulseless", score=0.42)]),
                             0.12, 0.42, min_relevance=0.60)
    assert "0.42" in detail and "0.60" in detail      # semantic arm
    assert "0.12" in detail and "0.18" in detail      # lexical arm
    assert "vt_vf.md#Pulseless" in detail
    # The similarity arm is only meaningful under BGE — silence about that sends the reader to tune
    # a threshold that cannot work on the hashing embedder.
    assert "EMBEDDER=bge" in detail


def test_detail_distinguishes_a_correct_decline_from_a_gate_that_is_too_strict():
    detail = explain_decline(_result([Passage(text="…", source="x.md", score=0.1)]), 0.0, 0.1)
    assert "ON-topic" in detail and "KB_MIN_RELEVANCE" in detail
    assert "the decline was correct" in detail


# --- the gate itself ----------------------------------------------------------------------------

def _p(score):
    return [Passage(text="…", source="vt_vf.md#Stable", score=score)]


def test_semantic_similarity_alone_is_enough():
    # The fix: a passage that MEANS the same thing is answerable even when it shares almost no
    # words with the question. Measured on the corpus, this is the case the old gate refused.
    assert is_relevant(_result(_p(0.69)), overlap=0.12, top_score=0.69, min_relevance=0.60)


def test_word_overlap_alone_is_still_enough():
    # Keeps the hashing embedder working: its similarity scores don't separate on- from off-topic,
    # so the lexical arm is the only signal it has.
    assert is_relevant(_result(_p(0.31)), overlap=0.40, top_score=0.31, min_relevance=0.60)


def test_neither_signal_declines():
    assert not is_relevant(_result(_p(0.47)), overlap=0.0, top_score=0.47, min_relevance=0.60)


def test_a_graph_fact_always_vouches():
    rel = [Relationship(fact="afib CO_MORBID_WITH htn", source="graph")]
    assert is_relevant(_result(relationships=rel), overlap=0.0, top_score=0.0, min_relevance=0.60)


def test_nothing_retrieved_declines():
    assert not is_relevant(_result(), overlap=0.0, top_score=0.0, min_relevance=0.60)
