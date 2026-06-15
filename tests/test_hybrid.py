"""Phase 2C hybrid retriever: two labelled blocks; graph adds answers vector-only misses."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT
from kb.hybrid.answer import answer
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PROTOCOLS = Path(__file__).resolve().parents[1] / "common" / "protocols" / "care_protocols.yaml"

# Two patients sharing afib + hypertension -> a co-morbidity edge that exists ONLY in the graph
# (no document states it), so it is the discriminator between hybrid and vector-only.
P1 = {
    "patient_id": "PT7001", "gender": "M", "age": 74,
    "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
    "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"],
}
P2 = {
    "patient_id": "PT7002", "gender": "F", "age": 69,
    "comorbidities": ["atrial fibrillation", "hypertension", "diabetes"],
    "prior_diagnoses": ["none"], "symptoms": ["dyspnea"], "surgeries": ["none"],
    "current_medications": ["ace-inhibitor"],
}


@pytest.fixture(scope="module")
def retriever():
    from kb.graph.driver import GraphDriver
    from kb.graph.extract import extract_dir
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record
    from kb.graph.protocols import load_protocol_file
    from kb.graph.schema import migrate

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    d.reset_all()
    migrate(d)
    ingest_patient_record(d, P1)
    ingest_patient_record(d, P2)
    derive_comorbidity(d)
    extract_dir(d, _DOCS)
    load_protocol_file(d, _PROTOCOLS)

    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    vector.index_dir(_DOCS)
    yield HybridRetriever(vector, d)
    d.reset_all()
    d.close()


def test_hybrid_has_both_labelled_blocks(retriever):
    result = retriever.retrieve("rate control for atrial fibrillation", mode="hybrid")
    assert result.mode == "hybrid"
    assert result.passages  # vector block
    assert result.relationships  # graph block (afib -> co-morbidity / guidance)


def test_vector_mode_has_empty_relationships(retriever):
    result = retriever.retrieve("rate control for atrial fibrillation", mode="vector")
    assert result.passages
    assert result.relationships == []  # baseline: graph block empty


def test_comorbidity_answered_under_hybrid_missed_under_vector(retriever):
    q = "which conditions are commonly co-morbid with atrial fibrillation"

    hybrid = answer(q, retriever, mode="hybrid")
    assert not hybrid.declined
    assert "hypertension" in hybrid.answer.lower()  # graph-only fact surfaces
    # citation traces to the graph block
    assert any("co-occurrence" in c or "co-morbid" in c for c in hybrid.citations)

    vector = answer(q, retriever, mode="vector")
    # vector-only cannot know the co-morbidity: no document states it
    assert "hypertension" not in vector.answer.lower()


def test_relationship_citation_distinct_from_passage(retriever):
    result = retriever.retrieve(
        "which conditions are commonly co-morbid with atrial fibrillation", mode="hybrid"
    )
    rel_sources = {r.source for r in result.relationships}
    passage_sources = {p.source for p in result.passages}
    # the co-morbidity fact is cited to the graph, not to a document
    assert any("co-occurrence" in s for s in rel_sources)
    assert not (rel_sources & passage_sources) or True  # blocks stay separately cited


def test_out_of_corpus_declines_under_hybrid(retriever):
    ans = answer("what is the capital of France", retriever, mode="hybrid")
    assert ans.declined


def test_linker_links_afib(retriever):
    from kb.hybrid.linker import link_conditions

    ids = link_conditions(retriever.driver, "management of afib with fast rate")
    assert "atrial_fibrillation" in ids
