"""Query coverage review (2026-10-14): 78 clinician questions run against the live stack; these are
the ones that were wrong or partial, now pinned. The called event is a VF (feb5…) on bed 2 for a
patient who also has a NEWER VT — answers about "this event" must stay on the VF.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM
from common.schemas import ChatTurn, Passage, RetrievalResult
from kb.graph.lookup import asks_event_protocol, is_patient_specific, match_intent
from orchestrator.guardrails import UNGROUNDED_ANSWER, ungrounded_identifiers
from orchestrator.orchestrator import (
    _PATIENT_HINT,
    Orchestrator,
    _answer_action_items,
    _answer_clinical_history,
    _answer_event_count,
    _answer_event_summary,
    _answer_mews_meaning,
    _missing_topic,
    _speakable,
    _spoken_record,
    _spoken_summary,
)

NOW = 1_000_000.0
EV, PT = "feb5d443", "PT998224"


def _intent(q):
    return match_intent(q, now=NOW, patient_ref=PT, event_ref=EV)


# --- routing: the called event, never the patient's newest one ------------------------------------

@pytest.mark.parametrize("q, focus", [
    ("What rhythm was detected?", "rhythm"),
    ("How confident is the model about this rhythm?", "confidence"),
    ("When did this event happen?", "time"),
    ("Is this a false positive?", "false_positive"),
    ("Why was I alerted?", "why"),
    ("What is the criticality of this event?", "criticality"),
])
def test_questions_about_the_called_event(q, focus):
    assert _intent(q) == ("event_summary", {"event_uuid": EV, "focus": focus})


@pytest.mark.parametrize("q, name", [
    ("What should I do now?", "action_items_for_event"),
    ("What are the action items for this event?", "action_items_for_event"),
    ("Are there any outstanding action items?", "outstanding_action_items"),  # ward-wide
    ("Show me the ECG strip for this event", "ecg_strip_for_event"),
    ("How was the heart rate trending?", "hr_trend_for_event"),
])
def test_event_scoped_lookups(q, name):
    assert _intent(q)[0] == name


def test_an_emergency_is_not_an_action_item_lookup():
    assert _intent("The patient is coding, what do I do?") is None


@pytest.mark.parametrize("q, expected", [
    ("Any prior surgeries?", ("patient_clinical_history", {"patient_id": PT, "focus": "surgeries"})),
    ("How old is the patient?", ("patient_clinical_history",
                                 {"patient_id": PT, "focus": "demographics"})),
    ("Which bed is this patient on?", ("patient_bed", {"patient_id": PT})),
])
def test_patient_questions(q, expected):
    assert _intent(q) == expected


def test_age_patterns_stay_a_cohort_question():
    assert _intent("Any patterns between age and event type?") == ("cohort_patterns", {})


def test_how_many_events_today_is_a_count():
    name, params = _intent("How many events were there today?")
    assert name == "event_count_since" and params["window"] == "today"
    assert params["since"] == NOW - NOW % 86400


@pytest.mark.parametrize("q, expected", [
    ("What is the protocol for this event?", True),
    ("How do I treat this patient?", True),
    ("How do you treat sepsis?", False),             # names its own topic
    ("What is the protocol to handle ventricular tachycardia?", False),
])
def test_protocol_questions_about_the_called_event(q, expected):
    assert asks_event_protocol(q, EV) is expected


def test_patient_specific_questions_are_recognised():
    assert is_patient_specific("How confident is the model about this rhythm?")
    assert is_patient_specific("what does PT4543 have")
    assert not is_patient_specific("How do you treat sepsis?")


# --- answers ----------------------------------------------------------------------------------------

_WHY = ("Shown because: ventricular fibrillation (69% ≥ 60%), Critical; vitals also abnormal: "
        "MEWS 9 ≥ 3 (Heart Rate 3); HR rising +35 bpm over 65 min, SpO₂ falling -3.4 % over 32 min.")
_EV_ROW = {"patient": PT, "bed": "Unit1-Bed02", "event_type": "VENTRICULAR_FIBRILLATION",
           "confidence": 0.6934, "false_positive": False, "criticality": "Critical",
           "mews_risk": "Critical", "why": _WHY, "ts": NOW,
           "why_json": json.dumps({"vitals": {"mews": {"score": 9, "risk": "Critical"}}})}


def test_event_summary_answers():
    assert "69 percent confident the rhythm is ventricular fibrillation" in \
        _answer_event_summary(_EV_ROW, "confidence")
    assert "no. The model detected ventricular fibrillation" in \
        _answer_event_summary(_EV_ROW, "false_positive")
    fp = _answer_event_summary({**_EV_ROW, "event_type": "NORMAL_SINUS", "false_positive": True},
                               "false_positive")
    assert "yes" in fp and "false positive" in fp
    why = _answer_event_summary(_EV_ROW, "why")
    assert "you were alerted because" in why and "≥" not in why and "minutesute" not in why
    assert "rated critical, MEWS 9 (Critical)" in _answer_event_summary(_EV_ROW, "criticality")


def test_speakable_reads_units_as_words():
    assert _speakable("+35 bpm over 65 min") == "+35 beats per minute over 65 minutes"


def test_action_items_and_counts():
    rows = [{"patient": PT, "bed": "Unit1-Bed02", "action": "Critical: VF detected — initiate ACLS",
             "priority": "high", "status": "outstanding"}]
    assert _answer_action_items(rows).startswith(
        "For patient PT998224 on bed 2, the recommended actions are: Critical: VF detected: initiate")
    assert _answer_action_items([]) == "No action items are recorded for this event."
    assert _answer_event_count([{"events": 8, "patients": 6, "alerted": 7, "pending": 5}], "today") \
        == "8 events today for 6 patients: 7 alerted, 5 still awaiting acknowledgment."


def test_mews_meaning_uses_the_scorer_bands():
    assert _answer_mews_meaning(5, None) == "A MEWS of 5 is high risk."
    assert _answer_mews_meaning(9, "Escalate now.").endswith("critical risk. Escalate now.")


def test_demographics_answer():
    rows = [{"patient": PT, "age": 87, "gender": "M"}]
    assert _answer_clinical_history(rows, "demographics") == "Patient PT998224 is 87 years old, male."


def test_trend_and_strip_are_said_not_listed():
    trend = _spoken_record({"patient": PT, "bed": "Unit1-Bed02", "event": "VENTRICULAR_FIBRILLATION",
                            "hr_history": [156.0, 170.0, 194.0, 190.0]})
    assert "rising, from 156 to 190, peaking at 194" in trend and "170" not in trend
    strip = _spoken_record({"patient": PT, "bed": "Unit1-Bed02", "event": "VENTRICULAR_FIBRILLATION",
                            "ecg_plot": "data/plots/x.png"})
    assert strip.endswith("The ECG strip is available in the app.") and "png" not in strip


def test_ward_summaries():
    vt = _spoken_summary([{"patient": "PT1", "bed": "Unit1-Bed02"},
                          {"patient": "PT2", "bed": "Unit1-Bed06"}], "patients_with_event_type")
    assert vt == "2 patients: PT1 on bed 2 and PT2 on bed 6."
    items = _spoken_summary([{"bed": f"Unit1-Bed0{i}", "action": "x — y", "priority": "high"}
                             for i in range(1, 6)], "outstanding_action_items")
    assert items.startswith("5 outstanding action items across 5 beds, 5 high priority.")
    pat = _spoken_summary([{"event": "ATRIAL_FIBRILLATION", "gender": "F", "age_band": "65+", "n": 2},
                           {"event": "PVC", "gender": "M", "age_band": "<40", "n": 1}],
                          "cohort_patterns")
    assert "women aged 65 and over, 2 events" in pat and "men aged under 40, 1 event" in pat


# --- the free-text path: no patient data, no off-topic answers -----------------------------------

def test_off_topic_word_is_not_answered_from_a_nearby_passage():
    vt = [Passage(text="Treat pulseless VT with CPR and defibrillation.", source="vt_vf.md#VT",
                  score=0.61)]
    assert _missing_topic("How do you treat sepsis?", vt) == ["sepsis"]
    assert _missing_topic("How do you treat ventricular tachycardia?",
                          [Passage(text="ventricular tachycardia: CPR", source="vt_vf.md", score=.8)]) \
        == []


def test_citation_label_read_as_a_patient_is_blocked():
    assert ungrounded_identifiers("The patient's ID is #P1.", "[P1] (vt_vf.md) CPR") != []


class _State:
    def __init__(self):
        self.patient_ref, self.event_ref, self.turns = PT, None, []


class _Working:
    def __init__(self):
        self.s = _State()

    def get_or_create(self, _sid):
        return self.s

    def save(self, _s):
        pass

    def append_turn(self, _sid, turn: ChatTurn):
        self.s.turns.append(turn)


class _Hybrid:
    def __init__(self, passages):
        self.passages, self.calls = passages, 0

    def retrieve(self, q, mode="hybrid"):
        self.calls += 1
        return RetrievalResult(query=q, passages=self.passages)


class _LLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "Patient P3 has a confidence of 0.36."

    def embed(self, texts):
        return [[0.0] for _ in texts]


def _orch(passages):
    llm = _LLM()
    hybrid = _Hybrid(passages)
    orch = Orchestrator(working=_Working(), hybrid=hybrid,
                        episodic=SimpleNamespace(recall=lambda *a, **k: [], add=lambda *a, **k: None),
                        llm=DeidentifyingLLM(llm, RegexDeidentifier()), driver=None,
                        episodic_recall=False, min_relevance=0.0, llm_router=False)
    return orch, llm, hybrid


def test_patient_specific_question_with_no_query_gets_a_hint_not_generation():
    orch, llm, hybrid = _orch([])
    r = orch.handle_turn("s1", "Is the patient comfortable?", now=time.time())
    assert r.answer == _PATIENT_HINT and llm.prompts == [] and hybrid.calls == 0


def test_event_reports_never_reach_a_free_text_answer():
    report = Passage(text="Event report for P3, confidence 0.36", source="report:5307ac81#Class",
                     score=0.9)
    sop = Passage(text="Escalate a critical alarm within 15 minutes.",
                  source="critical_alarm_sop.md#Escalation", score=0.9)
    orch, llm, _ = _orch([report, sop])
    orch.handle_turn("s1", "when should I escalate a critical alarm?", now=time.time())
    assert "report:" not in llm.prompts[-1] and "0.36" not in llm.prompts[-1]
    assert "critical_alarm_sop.md" in llm.prompts[-1]


def test_unknown_topic_is_declined_not_answered():
    vt = Passage(text="Treat pulseless VT with CPR and defibrillation.", source="vt_vf.md#VT",
                 score=0.61)
    orch, llm, _ = _orch([vt])
    r = orch.handle_turn("s1", "How do you treat sepsis?", now=time.time())
    assert r.declined and llm.prompts == [] and r.answer != UNGROUNDED_ANSWER
