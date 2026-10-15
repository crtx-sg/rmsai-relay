"""Verifiable event listings (voice + chat): the right filter, stated in the answer, or a decline.

Observed 2026-10-12: "say all unacked events in bed 02" read out every event on the bed
(acknowledged and withheld ones too), "unacknowledged events on Unit1-Bed02" listed every bed, and
"all VT events" fell through to a model. Each answer now comes from `list_events` (the same rows as
the Events tab and `cli.events`) and says which filter it used.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM
from kb.graph.lookup import looks_like_event_listing, match_intent
from kb.graph.templates import describe_event_history, event_history_rows
from orchestrator.orchestrator import (
    _EVENT_LISTING_HINT, Orchestrator, _answer_event_history,
)
from tests.test_grounded_answers import _Hybrid, _ScriptedLLM, _Working

NOW = 1_000_000.0
H = "event_history"


@pytest.mark.parametrize("q, params", [
    ("say all unacked events in bed 02", {"status": "pending", "bed": "Unit1-Bed02"}),
    ("unacknowledged events on Unit1-Bed02", {"status": "pending", "bed": "Unit1-Bed02"}),
    ("list all unacknowledged events for this patient", {"status": "pending", "patient_id": "PT9"}),
    ("events not yet acknowledged for PT998224", {"status": "pending", "patient_id": "PT998224"}),
    ("acknowledged events on bed 2", {"status": "acknowledged", "bed": "Unit1-Bed02"}),
    ("withheld events", {"status": "withheld"}),
    ("all ventricular tachycardia events", {"status": "any", "event_type": "VENTRICULAR_TACHYCARDIA"}),
    ("show all alarms", {"status": "any"}),
    ("critical events for this patient",
     {"status": "any", "patient_id": "PT9", "min_criticality": "High"}),
    ("What is a critical event on bed five?",
     {"status": "any", "bed": "Unit1-Bed05", "min_criticality": "High"}),
    ("critical events for PT998224 in the last 6 hours",
     {"status": "any", "patient_id": "PT998224", "min_criticality": "High",
      "since": NOW - 6 * 3600, "window_hours": 6}),
    ("how many VT episodes in the last 6 hours",
     {"status": "any", "event_type": "VENTRICULAR_TACHYCARDIA", "since": NOW - 6 * 3600,
      "window_hours": 6}),
])
def test_listing_questions_carry_their_filter(q, params):
    assert match_intent(q, now=NOW, patient_ref="PT9") == (H, params)


@pytest.mark.parametrize("q, name", [
    ("which bed has the most severe alarm", "worklist"),
    ("which patient has the most alarms", "alarm_counts"),
    ("critical events in the last 6 hours", "critical_events_since"),
    ("what is the vitals at this event", "vitals_at_patient_last_event"),
])
def test_other_operational_intents_are_unchanged(q, name):
    assert match_intent(q, now=NOW, patient_ref="PT9")[0] == name


def test_singular_event_question_is_not_a_listing():
    # "the event" is not "all events": without a patient this is not answerable from the graph.
    assert match_intent("what were the vitals at the event?", now=NOW) is None


@pytest.mark.parametrize("q, declined", [
    ("list the events for Mr Smith", True),
    ("show events in the cardiac ward", True),
    ("list all events for this patient", True),       # no patient in session to scope it
    ("what are the alarm limits for SpO2", False),    # a document question
    ("what are the signs of AFib episodes", False),
    ("how should alarms be escalated", False),
])
def test_unresolvable_listings_are_detected(q, declined):
    assert match_intent(q, now=NOW) is None
    assert looks_like_event_listing(q) is declined


def test_filter_description():
    assert describe_event_history({"status": "pending", "bed": "bed 2", "window_hours": 6}) == (
        "pending (unacknowledged) events on bed Bed02 in the last 6 hours")
    assert describe_event_history({"status": "any", "event_type": "VENTRICULAR_TACHYCARDIA",
                                   "patient_id": "PT1", "min_criticality": "High"}) == (
        "High or Critical events of type ventricular tachycardia for patient PT1")


class _Driver:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def run_read(self, cypher, **params):
        self.calls.append(params)
        return [dict(r) for r in self.rows]


def _graph_row(eid, **kw):
    return {"id": eid, "patient": "PT998224", "bed": "Unit1-Bed02", "unit": "Unit1",
            "timestamp": 1_790_000_000.0, "event_type": "VENTRICULAR_TACHYCARDIA",
            "confidence": 0.52, "criticality": "Critical", "status": "reported",
            "alert_gate": True, **kw}


def test_rows_carry_the_shared_status():
    d = _Driver([_graph_row("a"), _graph_row("b", status="acknowledged"),
                 _graph_row("c", alert_gate=False)])
    rows = event_history_rows(d, bed="Unit1-Bed02")
    assert [r["status"] for r in rows] == ["pending", "acknowledged", "withheld"]
    assert rows[0]["event_id"] == "a" and rows[0]["event"] == "VENTRICULAR_TACHYCARDIA"


def test_text_answer_states_filter_and_lists_ids():
    rows = event_history_rows(_Driver([_graph_row("34924fb9-ca69"), _graph_row("6bbd8fa0-e240")]))
    out = _answer_event_history(rows, {"status": "pending", "bed": "Unit1-Bed02"})
    head, *lines = out.splitlines()
    assert head == "2 pending (unacknowledged) events on bed Unit1-Bed02 for 1 patient:"
    assert lines[0].endswith("id 34924fb9") and "Unit1-Bed02" not in lines[0]  # bed is the filter
    assert "VENTRICULAR TACHYCARDIA 52%" in lines[0] and "pending" in lines[0]


def test_listing_counts_patients_not_only_events():
    # "How many patients are assigned to my work list?" was answered "10 pending events": two of
    # them were one patient's VF and VT, so the answer must say 9 patients (2026-10-13 call-in).
    rows = event_history_rows(_Driver([
        _graph_row("a"), _graph_row("b", event_type="VENTRICULAR_FIBRILLATION"),
        _graph_row("c", patient="PT984532", bed="Unit1-Bed04")]))
    out = _answer_event_history(rows, {"status": "pending"}, spoken=True)
    assert out.startswith("3 pending (unacknowledged) events for 2 patients: ")
    assert _answer_event_history(rows, {"status": "pending"}).startswith(
        "3 pending (unacknowledged) events for 2 patients:\n")
    # a single event, or a listing already scoped to one patient, adds nothing
    assert "patient" not in _answer_event_history(rows[:1], {"status": "pending"}).splitlines()[0]


def test_empty_answer_names_the_filter():
    assert _answer_event_history([], {"status": "pending", "bed": "Unit1-Bed02"}) == (
        "No pending (unacknowledged) events on bed Unit1-Bed02.")


def test_spoken_answer_counts_and_reports_pending():
    rows = event_history_rows(_Driver([
        _graph_row("a"), _graph_row("b", event_type="VENTRICULAR_FIBRILLATION",
                                    status="acknowledged"),
        _graph_row("c", bed="Unit1-Bed05")]))
    out = _answer_event_history(rows, {"status": "any", "patient_id": "PT998224"}, spoken=True)
    assert out == ("3 events for patient PT998224: ventricular tachycardia twice, ventricular "
                   "fibrillation. Beds: bed 2, bed 5. 2 pending.")


def _orch(driver):
    inner = _ScriptedLLM("should not be called")
    orch = Orchestrator(working=_Working(), hybrid=_Hybrid(),
                        episodic=SimpleNamespace(recall=lambda *a, **k: [], add=lambda *a, **k: None),
                        llm=DeidentifyingLLM(inner, RegexDeidentifier()), driver=driver,
                        episodic_recall=False, min_relevance=0.0, llm_router=True)
    return orch, inner


def test_orchestrator_answers_listings_from_the_shared_query(capsys):
    orch, inner = _orch(_Driver([_graph_row("34924fb9-ca69")]))
    r = orch.handle_turn("s1", "say all unacked events in bed 02", now=NOW)
    assert r.answer.startswith("1 pending (unacknowledged) event on bed Unit1-Bed02:")
    assert inner.prompts == []  # no model call
    assert "[orchestrator] graph query event_history {'status': 'pending', 'bed': 'Unit1-Bed02'}" \
           " -> 1 row(s)" in capsys.readouterr().out


def test_orchestrator_declines_an_unresolvable_listing(capsys):
    orch, inner = _orch(_Driver([]))
    r = orch.handle_turn("s1", "list the events for Mr Smith", now=NOW)
    assert r.answer == _EVENT_LISTING_HINT
    assert inner.prompts == []  # neither the LLM router nor the KB model was asked
    assert "declined with a hint" in capsys.readouterr().out
