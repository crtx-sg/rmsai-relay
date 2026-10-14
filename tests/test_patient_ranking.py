"""Patients ranked sickest first: rounds / triage lists. Open alerts first (by their most severe
criticality), then the latest MEWS score; bed order on request; short lines (user decision
2026-10-14)."""

from __future__ import annotations

import json

import pytest

from kb.graph.lookup import match_intent
from kb.graph.templates import TEMPLATE_DEFAULTS
from orchestrator.orchestrator import _answer_patient_ranking

NOW = 1_000_000.0
S = {"now": NOW, "patient_ref": "PT1", "event_ref": "E1"}


@pytest.mark.parametrize("q", [
    "list patients by criticality", "List the patients in order of criticality",
    "which patients are the most critical", "who should I see first",
    "give me the list for bed rounds", "I'm doing my rounds, who's sickest",
    "rank the patients by severity", "sort patients by MEWS", "triage list please",
    "show me the sickest patients", "prioritise my patients", "list all patients and their criticality"])
def test_ranking_phrasings(q):
    assert match_intent(q, **S) == ("patients_by_criticality", {})
    assert match_intent(q, now=NOW) == ("patients_by_criticality", {})  # text chat too


def test_bed_order_on_request():
    assert match_intent("patients for ward rounds in bed order", **S) == (
        "patients_by_criticality", {"order": "bed"})


@pytest.mark.parametrize("q, name", [
    ("What is the criticality of this event?", "event_summary"),
    ("which bed has the most severe alarm", "worklist"),
    ("which patient has the most alarms", "alarm_counts"),
    ("which patients have reported ventricular tachycardia", "patients_with_event_type")])
def test_neighbouring_questions_keep_their_queries(q, name):
    assert match_intent(q, **S)[0] == name


def test_cpr_rounds_is_not_a_ward_round():
    assert match_intent("rounds of CPR for VF", **S) is None


def test_open_alert_states_use_the_closed_statuses():
    assert TEMPLATE_DEFAULTS["patients_by_criticality"]["closed"] == ["acknowledged", "resolved"]


def _row(patient, bed, open_crit=None, score=0, open_alerts=0, event="PAC", ts=1.0):
    return {"patient": patient, "bed": bed, "open_alerts": open_alerts,
            "open_criticality": open_crit, "open_event": event if open_alerts else None,
            "latest_event": event, "mews_risk": "Low", "ts": ts,
            "why_json": json.dumps({"vitals": {"mews": {"score": score}}})}


ROWS = [  # arrive in the template's order; the score re-sorts within a band
    _row("PT2", "Unit1-Bed02", "Critical", score=4, open_alerts=1, event="VENTRICULAR_FIBRILLATION"),
    _row("PT7", "Unit1-Bed07", "Critical", score=9, open_alerts=2, event="VENTRICULAR_TACHYCARDIA"),
    _row("PT9", "Unit1-Bed09", "High", score=5, open_alerts=1, event="SVT"),
    _row("PT1", "Unit1-Bed01", score=6),  # no open alerts: after every open one, despite MEWS 6
]


def test_sickest_first_and_spoken_lines():
    out = _answer_patient_ranking(ROWS, spoken=True)
    assert out.startswith("4 patients, sickest first. 1. Bed 7, PT7: critical, ventricular "
                          "tachycardia, MEWS 9, 2 open alerts. 2. Bed 2, PT2: critical, ventricular "
                          "fibrillation, MEWS 4, 1 open alert. 3. Bed 9, PT9: high, SVT, MEWS 5")
    assert out.endswith("4. Bed 1, PT1: no open alerts, latest PAC, MEWS 6.")


def test_bed_order_and_text_lines():
    out = _answer_patient_ranking(ROWS, order="bed")
    lines = out.splitlines()
    assert lines[0] == "4 patients, in bed order:"
    assert [line.split(",")[0] for line in lines[1:]] == [
        "1. Bed Unit1-Bed01", "2. Bed Unit1-Bed02", "3. Bed Unit1-Bed07", "4. Bed Unit1-Bed09"]


def test_a_long_phone_list_is_cut_with_a_count():
    rows = [_row(f"PT{i}", f"Unit1-Bed{i:02d}") for i in range(1, 15)]
    out = _answer_patient_ranking(rows, spoken=True)
    assert "10. " in out and "11. " not in out and out.endswith("And 4 more; ask for the rest by bed.")


def test_no_patients():
    assert _answer_patient_ranking([]) == "No patients have recorded events."
