"""Offline tests for spoken-query normalization (STT phrasing -> canonical text)."""

from __future__ import annotations

import pytest

from kb.graph.spoken import normalize_spoken_query as N


@pytest.mark.parametrize(
    "spoken,expected",
    [
        # spelled-out acronyms collapse; multi-letter words are untouched
        ("A V Block", "AV Block"),
        ("S V T event", "SVT event"),
        ("E C G strips", "ECG strips"),
        ("H R and B P trend", "HR and BP trend"),
        ("S T elevation", "ST elevation"),
        ("R B B B", "RBBB"),
        # number-word bed labels rebuild to the canonical "Unit{u}-Bed{bb}"
        ("on bed unit one bed oh one", "on bed Unit1-Bed01"),
        ("the patient in bed one", "the patient in bed Unit1-Bed01"),
        ("bed oh five", "bed Unit1-Bed05"),
        # spoken counts before time units become digits (additive)
        ("the last twenty four hours", "the last 24 hours"),
        ("the last thirty minutes", "the last 30 minutes"),
        ("the last ten minutes", "the last 10 minutes"),
    ],
)
def test_normalize_spoken(spoken, expected):
    assert N(spoken) == expected


@pytest.mark.parametrize(
    "typed",
    [
        "what is the status of events on Bed Unit1-Bed01?",
        "show all patients with an SVT event",
        "critical events in the last 24 hours",
        "list outstanding action items",
    ],
)
def test_typed_queries_unchanged(typed):
    assert N(typed) == typed   # near-no-op: digit labels + multi-letter words pass through
