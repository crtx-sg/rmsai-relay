"""Bed assignment is graph-authoritative: no patient on two beds, no bed with two patients.

Live 2026-10-09: the in-memory stub reset on every consumer restart, spreading one patient's
events over two beds and putting three patients on Bed04.
"""

from __future__ import annotations

import random
import re

import pytest

import orchestrator.patient_bootstrap as pb
from common.bed_assignment import BedAssignmentStub


class _Graph:
    """Just enough of the graph: patients and their current bed."""

    def __init__(self):
        self.bed: dict[str, str | None] = {}

    def run_read(self, cypher, **params):
        if "Patient {id:$id}" in cypher:
            pid = params["id"]
            if pid not in self.bed:
                return []
            b = self.bed[pid]
            return [{"id": pid, "bed": b, "unit": b.split("-")[0] if b else None}]
        return [{"id": p, "bed": b, "unit": b.split("-")[0]} for p, b in self.bed.items() if b]

    def run_write(self, cypher, **params):
        self.bed[params["pid"]] = params["bed"]

    def reset(self):
        self.bed.clear()

    def occupants(self):
        out: dict[str, list[str]] = {}
        for p, b in self.bed.items():
            if b:
                out.setdefault(b, []).append(p)
        return out


@pytest.fixture
def graph(monkeypatch):
    g = _Graph()
    monkeypatch.setattr(pb, "ingest_patient_record",
                        lambda d, history, bed=None: g.bed.__setitem__(history["patient_id"], bed[1]))
    monkeypatch.setattr(pb, "PatientHistoryStub",
                        lambda: type("H", (), {"get": staticmethod(lambda pid: type(
                            "R", (), {"to_dict": staticmethod(lambda: {"patient_id": pid})})())})())
    return g


def test_known_patient_keeps_the_bed_after_a_restart(graph):
    pb.ensure_patient(graph, BedAssignmentStub(), "PT1")
    pb.ensure_patient(graph, BedAssignmentStub(), "PT2")
    fresh = BedAssignmentStub()  # consumer restart: empty in-memory stub
    assert pb.ensure_patient(graph, fresh, "PT2") == ("Unit1", "Unit1-Bed02")
    assert pb.ensure_patient(graph, fresh, "PT3") == ("Unit1", "Unit1-Bed03")  # not Bed01/02


def test_stale_stub_after_a_graph_reset_never_double_books(graph):
    beds = BedAssignmentStub()
    pb.ensure_patient(graph, beds, "PT_A")         # PT_A -> Bed01 (stub remembers it)
    graph.reset()                                   # graph wiped under a running consumer
    assert pb.ensure_patient(graph, beds, "PT_B")[1] == "Unit1-Bed01"   # free in the graph
    assert pb.ensure_patient(graph, beds, "PT_A")[1] == "Unit1-Bed02"   # not the stale Bed01


def test_known_patient_without_a_bed_gets_a_free_one(graph):
    graph.bed.update({"PT1": "Unit1-Bed01", "PT2": None})
    assert pb.ensure_patient(graph, BedAssignmentStub(), "PT2") == ("Unit1", "Unit1-Bed02")
    assert graph.bed["PT2"] == "Unit1-Bed02"


def test_random_runs_with_restarts_and_resets_never_overlap(graph):
    rng = random.Random(7)
    beds = BedAssignmentStub()
    seen_bed: dict[str, str] = {}
    for step in range(400):
        if rng.random() < 0.05:
            beds = BedAssignmentStub()             # consumer restart
        if rng.random() < 0.01:
            graph.reset()
            seen_bed.clear()
        pid = f"PT{rng.randrange(40)}"
        _, bed = pb.ensure_patient(graph, beds, pid)
        assert seen_bed.setdefault(pid, bed) == bed, f"{pid} moved beds at step {step}"
        assert all(len(ps) == 1 for ps in graph.occupants().values()), graph.occupants()
    assert all(re.fullmatch(r"Unit\d+-Bed\d{2}", b) for b in graph.occupants())
