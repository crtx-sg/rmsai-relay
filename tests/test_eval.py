"""Phase 2D evaluation harness: hybrid beats vector on relationship questions."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT
from kb.eval.harness import load_questions, render_report, run_eval

_QUESTIONS = Path(__file__).resolve().parents[1] / "kb" / "eval" / "questions.json"


def test_questions_load():
    qs = load_questions(_QUESTIONS)
    assert len(qs) >= 5
    assert {q.type for q in qs} == {"relationship", "passage", "safety"}


@pytest.fixture(scope="module")
def report():
    from kb.eval.fixtures import seed_eval_graph
    from kb.graph.driver import GraphDriver
    from kb.hybrid.retriever import HybridRetriever
    from kb.vector.retriever import VectorRetriever
    from kb.vector.store import QdrantStore

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    seed_eval_graph(d)
    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    vector.index_dir(Path(__file__).resolve().parents[1] / "docs")
    rep = run_eval(HybridRetriever(vector, d), load_questions(_QUESTIONS))
    yield rep
    d.reset_all()
    d.close()


@pytest.mark.infra
def test_hybrid_beats_vector_on_relationships(report):
    v = report["modes"]["vector"]["correctness_by_type"]["relationship"]
    h = report["modes"]["hybrid"]["correctness_by_type"]["relationship"]
    assert h > v
    assert h == pytest.approx(1.0)  # hybrid answers all relationship questions
    assert v == 0.0  # vector-only misses them all


@pytest.mark.infra
def test_passage_questions_answered_by_both(report):
    v = report["modes"]["vector"]["correctness_by_type"]["passage"]
    h = report["modes"]["hybrid"]["correctness_by_type"]["passage"]
    assert v == pytest.approx(1.0) and h == pytest.approx(1.0)


@pytest.mark.infra
def test_safety_questions_declined_by_both(report):
    for mode in ("vector", "hybrid"):
        assert report["modes"][mode]["correctness_by_type"]["safety"] == pytest.approx(1.0)


@pytest.mark.infra
def test_hybrid_costs_more_context(report):
    # The graph block adds context — hybrid should carry at least as many tokens as vector.
    assert report["modes"]["hybrid"]["avg_tokens"] >= report["modes"]["vector"]["avg_tokens"]


@pytest.mark.infra
def test_grounding_reported(report):
    assert 0.0 <= report["modes"]["hybrid"]["grounding"] <= 1.0
    assert render_report(report).startswith("KB evaluation")
