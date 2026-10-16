"""Every vital's history is stored on the event, and "last N <vital> readings" answers from it.

Offline (no Neo4j): the graph write is captured, the routing and the answer are pure functions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from kb.graph.lookup import match_intent
from orchestrator.orchestrator import _answer_vital_readings

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _ForcedModel(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.91


def test_every_vital_history_reaches_the_graph_write(monkeypatch):
    import orchestrator.event_flow as ef

    captured = {}

    class _Stop(Exception):
        pass

    def fake_persist(driver, **kw):
        captured.update(kw)
        raise _Stop  # only the write's arguments matter here

    monkeypatch.setattr(ef, "persist_monitored_event", fake_persist)
    event = process_window(next(read_hdf5_file(_FIXTURE)), _ForcedModel(), MewsVitalsAnalysis())
    with pytest.raises(_Stop):
        ef.process_device_event(event, None, None, bed=("Unit1", "Unit1-Bed05"))
    extra = captured["extra"]
    for field in ("hr", "sbp", "dbp", "spo2", "rr", "temp"):
        assert extra[f"{field}_history"]
        assert len(extra[f"{field}_history"]) == len(extra[f"{field}_history_ts"])


@pytest.mark.parametrize("query, expected", [
    # live call 2026-10-16: answered with the single snapshot at the event
    ("Give me the last three respiratory readings for this patient", {"n": 3, "focus": "rr"}),
    ("Give me last Three Vitals, three vital readings for this patient", {"n": 3}),
    ("last 5 blood pressure readings", {"n": 5, "focus": "bp"}),
    ("show the oxygen readings", {"n": 3, "focus": "spo2"}),
])
def test_readings_route_to_the_history_query(query, expected):
    assert match_intent(query, now=0, patient_ref="PT1", event_ref="e1") == (
        "vital_readings_for_event", {"event_uuid": "e1", **expected})


@pytest.mark.parametrize("query", [
    "What are the vitals at the time of this event?",
    "What was the blood pressure reported for this patient?",
    "what is the respiratory rate",
])
def test_snapshot_questions_still_get_the_snapshot(query):
    assert match_intent(query, now=0, patient_ref="PT1", event_ref="e1")[0] == \
        "vitals_at_selected_event"


_ROW = {"patient": "PT1", "bed": "Unit1-Bed01", "rr": [14.0, 15.3, 15.0, 15.8],
        "sbp": [120.0, 124.0], "dbp": [80.0], "hr": None}


def test_answer_gives_the_last_n_of_the_asked_vital():
    out = _answer_vital_readings(_ROW, "rr", 3)
    assert "last 3 readings" in out and "respiratory rate 15, 15, 16" in out
    assert "heart rate" not in out


def test_answer_says_when_a_history_is_missing():
    out = _answer_vital_readings(_ROW, None, 2)
    assert "heart rate not recorded" in out
    assert "systolic blood pressure 120, 124" in out and "diastolic blood pressure 80" in out
