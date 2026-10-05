"""Answers stay on the right patient (observed 2026-10-05: with PT992591's event selected, "what is the
MEWS score?" went to document retrieval and the model answered about PT4543, the example id in its
own instructions).

Three layers, each tested offline: MEWS questions route to the selected event's data; the MEWS answer
is rendered from the stored explanation; and an LLM answer naming a patient it was never given is
replaced.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM
from common.schemas import ChatTurn
from kb.graph.lookup import match_intent
from orchestrator.guardrails import UNCONFIRMED_PATIENT, foreign_patient_refs
from orchestrator.orchestrator import _ANSWER_INSTRUCTIONS, Orchestrator, _answer_operational

NOW = 1_000_000.0
EVT = "231a9c9a-e5e6-58ff-97f2-3283dad269e0"

# --- 1. routing -----------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    "what is the MEWS score?", "MEWS?", "what's the early warning score", "what is the criticality",
    "what is the news score",          # STT's rendering of "MEWS score"
    "is the NEWS 5 correct",
])
def test_mews_questions_about_the_selected_event_use_its_data(q):
    assert match_intent(q, now=NOW, patient_ref="PT992591", event_ref=EVT) == (
        "mews_at_selected_event", {"event_uuid": EVT})


def test_without_a_selected_event_the_session_patient_is_used():
    assert match_intent("what is the MEWS score?", now=NOW, patient_ref="PT992591") == (
        "mews_at_patient_last_event", {"patient_id": "PT992591"})


@pytest.mark.parametrize("q", [
    "how is MEWS calculated?", "what does a MEWS of 5 mean?", "when should MEWS escalate?",
    "explain the MEWS criteria",
])
def test_questions_about_mews_itself_stay_with_the_documents(q):
    assert match_intent(q, now=NOW, patient_ref="PT992591", event_ref=EVT) is None


def test_no_scope_no_lookup():
    assert match_intent("what is the MEWS score?", now=NOW) is None


def test_everyday_news_is_not_mews():
    assert match_intent("any news on this patient?", now=NOW, patient_ref="PT992591",
                        event_ref=EVT) != ("mews_at_selected_event", {"event_uuid": EVT})


# --- 2. rendering ---------------------------------------------------------------------------------

_WHY = json.dumps({"vitals": {"mews": {"score": 6, "risk": "High", "threshold": 3, "components": [
    {"name": "Respiratory Rate", "value": 25, "score": 2}, {"name": "SpO2", "value": 89, "score": 2},
    {"name": "Heart Rate", "value": 105, "score": 1}]}}})


def test_mews_answer_comes_from_the_stored_explanation():
    row = {"patient": "PT992591", "bed": "Unit1-Bed03", "event_type": "VENTRICULAR_TACHYCARDIA",
           "criticality": "Critical", "mews_risk": "High", "why_json": _WHY}
    out = _answer_operational([row])
    assert "patient PT992591" in out and "MEWS 6 (High)" in out
    assert "Respiratory Rate 25 scores 2, SpO2 89 scores 2, Heart Rate 105 scores 1" in out
    assert "{" not in out and "why" not in out  # the raw JSON is never voiced


@pytest.mark.parametrize("why", [None, "", "not json", json.dumps({"vitals": {}})])
def test_missing_or_broken_explanation_falls_back_to_the_risk(why):
    out = _answer_operational([{"patient": "PT992591", "mews_risk": "High", "why_json": why}])
    assert "MEWS risk High" in out and "{" not in out


# --- 3. output guardrail --------------------------------------------------------------------------


def test_prompt_carries_no_example_patient_id():
    assert foreign_patient_refs(_ANSWER_INSTRUCTIONS, "") == []


def test_foreign_patient_refs():
    ctx = "## Known relationships\n- PT992591 had VT"
    assert foreign_patient_refs("PT992591 has MEWS 6", ctx) == []
    assert foreign_patient_refs("PT4543 has MEWS 0", ctx) == ["PT4543"]
    assert foreign_patient_refs("pt992591 is stable", ctx) == []          # case-insensitive
    assert foreign_patient_refs("PT930787 is stable", "", "PT930787") == []  # the session patient


class _State:
    def __init__(self):
        self.patient_ref = "PT992591"
        self.event_ref = None
        self.turns: list[ChatTurn] = []


class _Working:
    def __init__(self):
        self.s = _State()

    def get_or_create(self, _sid):
        return self.s

    def save(self, _s):
        pass

    def append_turn(self, _sid, turn):
        self.s.turns.append(turn)


class _Hybrid:  # one relationship, so the turn is not declined and the LLM runs
    def retrieve(self, _q, mode="hybrid"):
        return SimpleNamespace(passages=[], relationships=[
            SimpleNamespace(source="mews_escalation.md", fact="MEWS >= 5 means escalate")])


class _ScriptedLLM:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, prompt, **kwargs):
        return self.answer

    def embed(self, texts):
        return [[0.0] for _ in texts]


def _turn(answer):
    orch = Orchestrator(working=_Working(), hybrid=_Hybrid(),
                        episodic=SimpleNamespace(recall=lambda *a, **k: [], add=lambda *a, **k: None),
                        llm=DeidentifyingLLM(_ScriptedLLM(answer), RegexDeidentifier()),
                        driver=None, episodic_recall=False)
    return orch.handle_turn("s1", "when should I escalate?", now=NOW)


def test_answer_naming_an_unknown_patient_is_replaced(capsys):
    r = _turn("The patient PT4543 has a MEWS score of 0 (Low).")
    assert r.answer == UNCONFIRMED_PATIENT
    assert "blocked answer naming patient(s) not in context: PT4543" in capsys.readouterr().out
    assert any(span.get("blocked_patient_refs") == ["PT4543"] for span in r.trace)


def test_answer_about_the_session_patient_passes():
    r = _turn("Escalate PT992591 when MEWS reaches 5.")
    assert r.answer == "Escalate PT992591 when MEWS reaches 5."
