"""Regressions from the live outbound call of 2026-10-14 (PT998224, VF on bed 2).

The MEWS answer was computed but never heard (split question), a misheard bed was answered with the
call's own patient, MEWS for a named bed fell through, past medical history listed monitor events,
and "kindly acknowledge this" was not an acknowledgment. Phrases below are the ones from that call.
"""

from __future__ import annotations

import json
import time

import pytest

from kb.graph.lookup import match_intent, names_unresolved_bed
from orchestrator.orchestrator import _answer_clinical_history, _spoken_record
from voice.livekit_agent import is_dangling, join_carried
from voice.outbound import parse_ack

NOW = 1_000_000.0
EV, PT = "feb5d443", "PT998224"


def _intent(q):
    return match_intent(q, now=NOW, patient_ref=PT, event_ref=EV)


# --- split questions (worker) ----------------------------------------------------------------------

@pytest.mark.parametrize("piece", ["What was the", "What is the", "and the", "the MEWS of"])
def test_half_questions_are_held_for_the_rest(piece):
    assert is_dangling(piece)


@pytest.mark.parametrize("q", ["What was the MEWS score", "which bed is PT998224 on",
                               "Kindly acknowledge this", "What are the other events reported for",
                               # whole questions that end on "is/was/the" (review 2026-10-14)
                               "Can you tell me what the potassium was?",
                               "Do you know what the MEWS score is", "who is the patient this is about",
                               "Repeat what the alert was"])
def test_whole_questions_are_answered(q):
    assert not is_dangling(q)


def test_carried_piece_joins_the_next_one_and_routes_to_mews_for_the_bed():
    t = time.monotonic()
    joined = join_carried(("What was the MEWS score", t), "For the last reported event on bed "
                                                          "number seven", t + 2)
    assert joined == "What was the MEWS score For the last reported event on bed number seven"
    assert _intent(joined) == ("bed_last_event", {"bed": "Unit1-Bed07", "focus": "mews"})
    assert join_carried(("What was the", t), "later", t + 60) == "later"  # stale piece dropped


# --- routing ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", ["What was the last reported event on bed seven?",
                               "Okay. What was the last reported event on vid nine?"])
def test_last_event_on_a_bed_is_that_event_not_the_call(q):
    name, params = _intent(q)
    assert name == "bed_last_event" and params["bed"] in ("Unit1-Bed07", "Unit1-Bed09")


def test_unresolved_bed_is_not_guessed():
    assert _intent("what was the last event on bed nein") is None
    assert names_unresolved_bed("what was the last event on bed nein")


@pytest.mark.parametrize("q, focus", [
    ("What is the past medical history of this patient", None),
    ("what medications is the patient on", "medications"),
    ("does the patient have any allergies", "allergies"),
    ("any prior surgeries for this patient", "surgeries"),
    ("what are his diagnoses", "conditions"),
])
def test_clinical_history_questions(q, focus):
    assert _intent(q) == ("patient_clinical_history",
                          {"patient_id": PT, **({"focus": focus} if focus else {})})


def test_prior_events_stay_an_event_listing():
    assert _intent("any prior events for this patient")[0] == "event_history"


def test_mews_for_this_event_still_uses_the_call():
    assert _intent("Uh, news score for this event") == ("mews_at_selected_event", {"event_uuid": EV})


# --- spoken answers --------------------------------------------------------------------------------

_WHY = json.dumps({"vitals": {"mews": {"score": 9, "risk": "Critical", "components": [
    {"name": "Heart Rate", "value": 195, "score": 3}, {"name": "Systolic BP", "value": 43, "score": 3},
    {"name": "SpO2", "value": 84, "score": 3}, {"name": "Temp", "value": 99.5, "score": 0}]}}})
_ROW = {"patient": PT, "bed": "Unit1-Bed02", "event_type": "VENTRICULAR_FIBRILLATION",
        "criticality": "Critical", "mews_risk": "Critical", "why_json": _WHY, "hr": 195.0,
        "sbp": 43.0, "dbp": 30.0, "spo2": 84.0, "rr": 10.0, "temp": 99.5, "ts": NOW}


def test_mews_answer_gives_the_score_and_what_drove_it():
    out = _spoken_record(_ROW, "mews")
    assert "MEWS 9, critical risk" in out
    assert "systolic blood pressure 43 scores 3" in out and "temperature" not in out  # score-0 omitted


def test_vitals_answer_is_said_not_read():
    out = _spoken_record(_ROW)
    assert out.startswith("Patient PT998224 on bed 2: ventricular fibrillation")
    assert "blood pressure 43 over 30" in out and "oxygen saturation 84 percent" in out
    assert "S P O 2" not in out and "sbp" not in out and "VENTRICULAR" not in out


def test_one_vital_asked_one_vital_answered():
    out = _spoken_record(_ROW, "hr")
    assert out.endswith("Heart rate 195.") and "blood pressure" not in out


def test_clinical_history_expands_abbreviations():
    rows = [{"patient": PT, "age": 87, "gender": "M", "conditions": ["MI", "COPD"],
             "medications": ["statin"], "surgeries": ["valve repair", "pacemaker"],
             "symptoms": ["dyspnea"]}]
    out = _answer_clinical_history(rows)
    assert "87 year old male" in out and "myocardial infarction" in out
    assert "Medications: statin." in out and "valve repair and pacemaker" in out
    assert _answer_clinical_history(rows, "medications") == f"Patient {PT}. Medications: statin."
    assert "not recorded" in _answer_clinical_history(rows, "allergies")


# --- acknowledgment ----------------------------------------------------------------------------------

def test_kindly_acknowledge_this_is_an_ack():
    assert parse_ack("Kindly acknowledge this") == "yes"
