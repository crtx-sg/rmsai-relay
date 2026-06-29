"""Offline unit tests for `match_intent` — NL query -> (template, params), no graph needed.

The vitals intent is patient-scoped: it only fires when a `patient_ref` is in scope (the chat
session), resolving "the event" to that patient's most recent MonitoredEvent. Without a patient
(the operational CLI path) a bare vitals question must fall through, not mis-route.
"""

from __future__ import annotations

import pytest

from kb.graph.lookup import match_intent

NOW = 1_000_000.0


# The operational matrix (the README questions) -> (template, params). Each clinician question must
# route to its tested template on the inbound text path (no patient session required).
@pytest.mark.parametrize(
    "query,expected",
    [
        ("List all patients, bed, unit who had critical events in the last 24 hours",
         ("critical_events_since", {"since": NOW - 24 * 3600})),
        ("List positive patient events reported in the last 30 minutes",
         ("positive_events_since", {"since": NOW - 30 * 60})),
        ("What is the status of patient events reported on Bed Unit1-Bed01?",
         ("event_status_on_bed", {"bed": "Unit1-Bed01"})),
        ("Get the event analysis report for Bed Unit1-Bed01 in the unit",
         ("reports_for_bed", {"bed": "Unit1-Bed01"})),
        ("What were the vitals at the time of the event for the patient in Bed Unit1-Bed01?",
         ("vitals_for_bed_last_event", {"bed": "Unit1-Bed01"})),
        ("Provide an action item list of outstanding actions for patients",
         ("outstanding_action_items", {})),
        ("What is the care protocol for bed Unit1-Bed01 with the last reported event?",
         ("protocol_for_bed_last_event", {"bed": "Unit1-Bed01"})),
        ("From the data, any pattern of age, gender, co-morbidities or symptoms leading to an event?",
         ("cohort_patterns", {})),
        ("Show me the ECG strips for the patient with the last reported AFib event",
         ("ecg_strips_last_event_of_type", {"event_type": "ATRIAL_FIBRILLATION"})),
        ("Show me the HR and BP trend for the patient with the last reported Tachycardia event",
         ("trend_last_event_of_type", {"event_type": "SINUS_TACHYCARDIA"})),
        ("Show all patients who have an AFib event",
         ("patients_with_event_type", {"event_type": "ATRIAL_FIBRILLATION"})),
    ],
)
def test_operational_matrix_routes(query, expected):
    assert match_intent(query, now=NOW) == expected


# Spoken/STT phrasing must route the same as typed — acronyms spelled out, bed labels as number
# words. (match_intent normalizes the query first; see kb/graph/spoken.py.)
@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("show all patients with an S V T event",
         ("patients_with_event_type", {"event_type": "SVT"})),
        ("ECG strips for the last A V Block event",
         ("ecg_strips_last_event_of_type", {"event_type": "AV_BLOCK_1"})),
        ("what is the status of events on bed unit one bed oh one",
         ("event_status_on_bed", {"bed": "Unit1-Bed01"})),
        ("what were the vitals for the patient in bed one",
         ("vitals_for_bed_last_event", {"bed": "Unit1-Bed01"})),
        ("critical events in the last twenty four hours",
         ("critical_events_since", {"since": NOW - 24 * 3600})),
    ],
)
def test_spoken_queries_route_like_typed(spoken, expected):
    assert match_intent(spoken, now=NOW) == expected


def test_pattern_wins_over_comorbidity_when_both_present():
    # "pattern ... co-morbidities" must hit the cohort analytic, not the comorbidity-of-X traversal.
    assert match_intent("any pattern of co-morbidities leading to events?", now=NOW) == (
        "cohort_patterns", {})


def test_this_patient_scopes_to_session_patient():
    # With a patient in session, "this patient" scopes critical/all events to that patient.
    assert match_intent("what are other critical events for this patient?", now=NOW,
                        patient_ref="PT8620") == ("critical_events_for_patient",
                                                  {"patient_id": "PT8620"})
    assert match_intent("show this patient's events", now=NOW, patient_ref="PT8620") == (
        "events_for_patient", {"patient_id": "PT8620"})


def test_this_patient_without_session_falls_through_to_global():
    # No patient in session (e.g. text chat) -> "this patient" can't resolve; global critical query.
    assert match_intent("critical events for this patient", now=NOW) == (
        "critical_events_since", {"since": NOW - 24 * 3600})


def test_global_critical_not_hijacked_by_this_patient_rule():
    # A plain global query (no "this patient") stays global even with a patient in session.
    assert match_intent("list all patients with critical events in the last 24 hours", now=NOW,
                        patient_ref="PT8620") == ("critical_events_since",
                                                  {"since": NOW - 24 * 3600})


def test_comorbidity_traversal_still_works():
    name, params = match_intent("what are the comorbidities of atrial fibrillation?", now=NOW)
    assert name == "comorbidity_neighborhood"
    assert params["condition_id"] == "atrial_fibrillation"


# Event-scoped intents must carry the resolved event_type through as a parameter — not be hardwired
# to AFib/Tachycardia. Same question, different event name -> same template, different param.
@pytest.mark.parametrize(
    "phrase,event_type",
    [
        ("AFib", "ATRIAL_FIBRILLATION"),
        ("v-tach", "VENTRICULAR_TACHYCARDIA"),
        ("ST elevation", "ST_ELEVATION"),
        ("atrial flutter", "ATRIAL_FLUTTER"),
        ("SVT", "SVT"),
    ],
)
def test_event_scoped_intents_are_parameterized(phrase, event_type):
    assert match_intent(f"show all patients with a {phrase} event", now=NOW) == (
        "patients_with_event_type", {"event_type": event_type})
    assert match_intent(f"show the ECG strips for the last {phrase} event", now=NOW) == (
        "ecg_strips_last_event_of_type", {"event_type": event_type})
    assert match_intent(f"HR and BP trend for the last {phrase} event", now=NOW) == (
        "trend_last_event_of_type", {"event_type": event_type})


@pytest.mark.parametrize(
    "query",
    [
        "what were the vitals at the time of the event?",
        "what was the heart rate during the event",
        "show me the blood pressure",
        "what was the SpO2",
        "respiratory rate please",
        "what was the patient's temperature",
        "give me the BP and HR",
    ],
)
def test_vitals_intent_when_patient_scoped(query):
    intent = match_intent(query, now=NOW, patient_ref="PT7878")
    assert intent == ("vitals_at_patient_last_event", {"patient_id": "PT7878"})


def test_vitals_intent_requires_patient_scope():
    # No patient in scope (operational CLI path) -> vitals question must NOT route to the
    # patient-scoped template; it falls through to None (template/Cypher fallback handles it).
    assert match_intent("what were the vitals at the event?", now=NOW) is None
    assert match_intent("what were the vitals at the event?", now=NOW, patient_ref=None) is None


def test_non_vitals_query_not_hijacked_by_vitals_rule():
    # A patient is in scope, but the question is operational, not about vitals.
    assert match_intent("what are the critical events?", now=NOW, patient_ref="PT7878") == (
        "critical_events_since",
        {"since": NOW - 24 * 3600},
    )
    assert match_intent("any outstanding action items?", now=NOW, patient_ref="PT7878") == (
        "outstanding_action_items",
        {},
    )


def test_unknown_query_returns_none():
    assert match_intent("tell me a joke", now=NOW, patient_ref="PT7878") is None
