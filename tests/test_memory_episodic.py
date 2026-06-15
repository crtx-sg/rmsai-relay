"""Episodic memory: store -> recall (in-memory) + cross-instance recall (live Qdrant)."""

from __future__ import annotations

import uuid

import pytest

from common.config import DEFAULT
from memory.episodic import EpisodicMemory


# --- in-memory (no server) ---


def test_store_and_recall_in_memory():
    mem = EpisodicMemory.in_memory()
    mem.add("clinician acknowledged the VT alert on bed 3", session_id="s1", patient_ref="PT1")
    mem.add("discussed afib rate control with beta blockers", session_id="s1", patient_ref="PT1")
    hits = mem.recall("what did we say about atrial fibrillation rate control")
    assert hits
    assert "afib" in hits[0].text or "rate control" in hits[0].text


def test_recall_filters_by_patient():
    mem = EpisodicMemory.in_memory()
    mem.add("PT1 VT alert acknowledged", patient_ref="PT1")
    mem.add("PT2 afib discussion", patient_ref="PT2")
    hits = mem.recall("alert", patient_ref="PT1")
    assert hits and all(h.patient_ref == "PT1" for h in hits)


def test_add_is_idempotent_by_content():
    mem = EpisodicMemory.in_memory()
    a = mem.add("same episode text", session_id="s1", timestamp=1.0)
    b = mem.add("same episode text", session_id="s1", timestamp=1.0)
    assert a == b  # stable id -> upsert, not duplicate


# --- cross-instance recall against live Qdrant ---


@pytest.mark.infra
def test_cross_instance_recall_live_qdrant():
    pytest.importorskip("qdrant_client")
    collection = f"rmsai_episodic_test_{uuid.uuid4().hex[:8]}"
    try:
        writer = EpisodicMemory.from_config(DEFAULT, embedder_name="hashing", collection=collection)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qdrant unreachable: {exc}")
    try:
        writer.add("clinician acknowledged ventricular tachycardia alert", patient_ref="PT9")
        # a fresh instance ("another process") recalls it from the shared backend
        reader = EpisodicMemory.from_config(DEFAULT, embedder_name="hashing", collection=collection)
        hits = reader.recall("VT alert acknowledgement", patient_ref="PT9")
        assert hits and "ventricular tachycardia" in hits[0].text
    finally:
        writer.client.delete_collection(collection)
