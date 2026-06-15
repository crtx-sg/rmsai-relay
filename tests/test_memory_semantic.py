"""Semantic memory tier (= Phase 2A index): index + recall."""

from __future__ import annotations

from pathlib import Path

from memory.semantic import SemanticMemory

_DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_index_and_recall():
    mem = SemanticMemory.in_memory()
    assert mem.index_dir(_DOCS) > 0
    passages = mem.recall("rate control for atrial fibrillation", k=3)
    assert passages
    assert passages[0].source.startswith("afib_rvr.md")
