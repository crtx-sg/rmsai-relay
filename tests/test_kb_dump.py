"""Offline tests for the kb_dump harness + the vector append/reset index mode.

`render_dump` is a pure function (no stores). The append/reset + chunks_for_doc tests use an
in-memory Qdrant + the hashing embedder, so they need no live server.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cli.kb_dump import render_dump
from common.config import DEFAULT
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from orchestrator.event_flow import write_report

_DOCS = Path(__file__).resolve().parents[1] / "docs"

_GRAPH_ROW = {
    "patient": "PT4543",
    "event": {
        "event_type": "SVT", "criticality": "High", "status": "reported",
        "is_false_positive": False, "confidence": 1.0, "mews_risk": "Medium",
        "hr": 171.0, "sbp": 136.0, "dbp": 92.0, "spo2": 98.0, "rr": 18.0, "temp": 97.9,
        "signal_ref": "hdf5://PT4543/evt1", "ecg_plot_ref": None, "vitals_plot_ref": None,
    },
    "bed": "Unit1-Bed01",
    "conditions": ["SVT"],
    "actions": [],
    "report": {"id": "report:evt1", "index_status": "indexed", "uri": "reports/evt1.md",
               "summary": "SVT for PT4543, MEWS 4 (Medium), confidence 1.00"},
}


# --- render_dump (pure) ---


def test_render_dump_shows_graph_and_vector():
    chunks = [{"text": "Alert for patient PT4543. Detected SVT.", "source": "report:evt1#0",
               "doc_id": "report:evt1"}]
    out = render_dump("evt1", _GRAPH_ROW, chunks)
    assert "GRAPH (Neo4j)" in out and "VECTOR (Qdrant)" in out
    assert "patient    : PT4543" in out
    assert "HR 171" in out and "BP 136.0/92.0" in out
    assert "chunks: 1" in out
    assert "Alert for patient PT4543" in out


def test_render_dump_missing_event():
    out = render_dump("nope", None, [])
    assert "no MonitoredEvent with this id" in out
    assert "chunks: 0" in out


def test_render_dump_shows_report_file_when_present():
    out = render_dump("evt1", _GRAPH_ROW, [], report_text="# Report\n\nAlert for PT4543. SVT.")
    assert "REPORT FILE (markdown)" in out
    assert "Alert for PT4543. SVT." in out
    assert "exists" in out


def test_render_dump_flags_missing_report_file():
    out = render_dump("evt1", _GRAPH_ROW, [], report_text=None)
    assert "MISSING on disk" in out
    assert "REPORT FILE (markdown)" not in out


# --- write_report materialization ---


def test_write_report_materializes_file(tmp_path):
    cfg = replace(DEFAULT, report_dir=str(tmp_path))
    uri = write_report("# Report\n\nAlert for PT9 SVT.\n", "evt1", config=cfg)
    p = Path(uri)
    assert p.is_file() and p.name == "evt1.md"
    assert "Alert for PT9 SVT." in p.read_text(encoding="utf-8")
    write_report("# Report v2\n", "evt1", config=cfg)  # replay overwrites its own report
    assert "v2" in p.read_text(encoding="utf-8")


# --- vector store: append vs reset + chunks_for_doc ---


def _retriever():
    return VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing",
                                 rerank=False)


def test_index_append_preserves_added_event_report():
    r = _retriever()
    r.index_dir(_DOCS)                                   # initial corpus (reset default)
    r.add_document("report:evt1", "Alert for PT9 supraventricular tachycardia heart rate 171")
    r.index_dir(_DOCS, reset=False)                      # re-index docs WITHOUT reset
    assert r.store.chunks_for_doc("report:evt1")        # event report survived


def test_index_reset_wipes_added_event_report():
    r = _retriever()
    r.index_dir(_DOCS)
    r.add_document("report:evt1", "Alert for PT9 supraventricular tachycardia heart rate 171")
    r.index_dir(_DOCS, reset=True)                       # explicit rebuild
    assert r.store.chunks_for_doc("report:evt1") == []   # event report gone


def test_chunks_for_doc_unknown_doc_is_empty():
    r = _retriever()
    r.index_dir(_DOCS)
    assert r.store.chunks_for_doc("report:does-not-exist") == []


def test_append_with_mismatched_embedder_dim_raises():
    r = _retriever()
    r.index_dir(_DOCS)                                   # collection built at the hashing dim
    assert r.store.vector_dim() == r.embedder.dim

    class _BigEmbedder:  # simulate switching to a higher-dim embedder (e.g. hashing -> BGE)
        name = "fake-big"
        dim = r.embedder.dim + 128

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    r.embedder = _BigEmbedder()
    with pytest.raises(ValueError, match="vector dim"):
        r.index_dir(_DOCS, reset=False)                  # clear guard, not a cryptic Qdrant error
