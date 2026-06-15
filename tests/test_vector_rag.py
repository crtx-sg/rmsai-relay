"""Phase 2A vector RAG: chunking, embeddings, retrieve, rerank, grounded ask."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.interfaces import LLMProvider
from kb.vector.answer import answer
from kb.vector.chunking import chunk_dir, chunk_document
from kb.vector.embeddings import HashingEmbedder, get_embedder
from kb.vector.rerank import LexicalReranker
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore, SearchHit

_DOCS = Path(__file__).resolve().parents[1] / "docs"


# --- chunking ---


def test_chunk_document_splits_on_headings():
    text = "# Title\nintro\n\n## A\nalpha body\n\n## B\nbeta body"
    chunks = chunk_document(text, "doc.md")
    sources = [c.source for c in chunks]
    assert "doc.md#A" in sources and "doc.md#B" in sources
    assert any("alpha body" in c.text for c in chunks)


def test_chunk_dir_skips_readme():
    chunks = chunk_dir(_DOCS)
    assert chunks
    assert all(c.doc_id.lower() != "readme.md" for c in chunks)


# --- embeddings ---


def test_hashing_embedder_deterministic_and_normalized():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["atrial fibrillation rate control"])[0]
    b = emb.embed(["atrial fibrillation rate control"])[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5  # L2 normalized


def test_get_embedder_hashing():
    assert get_embedder("hashing").name.startswith("hashing")


# --- retrieve (in-memory Qdrant, deterministic hashing embedder) ---


@pytest.fixture(scope="module")
def retriever() -> VectorRetriever:
    r = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    n = r.index_dir(_DOCS)
    assert n > 0
    return r


def test_retrieve_finds_relevant_doc(retriever):
    result = retriever.retrieve("how to rate control atrial fibrillation with fast heart rate", k=3)
    assert result.mode == "vector"
    assert result.relationships == []  # vector mode: graph block empty
    assert result.passages
    assert result.passages[0].source.startswith("afib_rvr.md")


def test_retrieve_vt_vf_query(retriever):
    result = retriever.retrieve("defibrillation for ventricular fibrillation cardiac arrest", k=3)
    assert result.passages[0].doc_id == "vt_vf.md" if hasattr(result.passages[0], "doc_id") else True
    assert "vt_vf.md" in result.passages[0].source


def test_passages_carry_citations_and_scores(retriever):
    result = retriever.retrieve("MEWS escalation threshold", k=3)
    for p in result.passages:
        assert "#" in p.source or p.source.endswith(".md")
        assert isinstance(p.score, float)


# --- rerank improves ordering ---


def test_lexical_reranker_promotes_query_overlap():
    # Vector scores rank an off-topic hit first; lexical overlap should fix the order.
    hits = [
        SearchHit(text="unrelated content about plumbing", source="x", doc_id="x", score=0.9),
        SearchHit(text="atrial fibrillation rate control beta blocker", source="y", doc_id="y",
                  score=0.6),
    ]
    reranked = LexicalReranker(weight=0.8).rerank("atrial fibrillation rate control", hits)
    assert reranked[0].source == "y"  # the relevant one is now first


# --- grounded ask + decline ---


def test_ask_returns_grounded_cited_answer(retriever):
    ans = answer("how do I rate control atrial fibrillation", retriever)
    assert not ans.declined
    assert ans.citations
    assert any("afib_rvr.md" in c for c in ans.citations)


def test_ask_declines_out_of_corpus(retriever):
    ans = answer("what is the capital of France", retriever)
    assert ans.declined
    assert "don't have information" in ans.answer


class _EchoLLM(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "GROUNDED: " + prompt.split("Question:")[-1].strip()

    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_ask_uses_llm_when_provided(retriever):
    ans = answer("how do I rate control atrial fibrillation", retriever, llm=_EchoLLM())
    assert ans.answer.startswith("GROUNDED:")
    assert ans.citations
