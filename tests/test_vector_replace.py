"""Re-indexing a document replaces it: chunks of the old version must not stay searchable."""

from __future__ import annotations

from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore


def test_reupload_replaces_old_sections():
    r = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing", rerank=False)
    r.add_document("sop.md", "# SOP\n\n## Dose\n\nGive amiodarone 300 mg bolus.\n\n"
                             "## Extra\n\nOld appendix text.\n")
    r.add_document("other.md", "# Other\n\n## Notes\n\nUnrelated document stays.\n")
    r.add_document("sop.md", "# SOP\n\n## Dosing\n\nGive amiodarone 150 mg.\n")

    texts = " ".join(p["text"] for p in r.store.chunks_for_doc("sop.md"))
    assert "150 mg" in texts
    assert "300 mg" not in texts and "Old appendix" not in texts
    assert r.store.chunks_for_doc("other.md")  # other documents untouched
