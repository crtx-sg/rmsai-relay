"""Offline unit tests for `match_intent` — NL query -> (template, params), no graph needed.

The vitals intent is patient-scoped: it only fires when a `patient_ref` is in scope (the chat
session), resolving "the event" to that patient's most recent MonitoredEvent. Without a patient
(the operational CLI path) a bare vitals question must fall through, not mis-route.
"""

from __future__ import annotations

import pytest

from kb.graph.lookup import match_intent

NOW = 1_000_000.0


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
