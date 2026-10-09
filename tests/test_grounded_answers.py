"""Grounded answers on the clinical line (observed 2026-10-09, live call + replay).

With no patient data in the prompt, the local model invented "Patient ID 1234, bed Bayside 3",
"Bed 1 and Bed 2", "Ward 4 Bay 1 (Bed 11)" and a mangled event uuid; de-identification of the whole
prompt turned citation markers into <US_DRIVER_LICENSE> and "bin three" into <PERSON>; worklist
questions never reached the graph; "thank you" was searched in the KB. Each fix is pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM
from common.schemas import Passage, Relationship
from kb.graph.llm_router import CATALOGUE
from kb.graph.lookup import match_intent
from kb.graph.templates import CLOSED_STATUSES, WORKLIST_LEVELS, run_template
from orchestrator.guardrails import UNGROUNDED_ANSWER, ungrounded_identifiers
from orchestrator.orchestrator import Orchestrator, _answer_operational

NOW = 1_000_000.0

# --- A. grounding guard ---------------------------------------------------------------------------

_CTX = ("## Matching records\n- patient PT992591, bed Unit1-Bed03, unit Unit1, event VT, "
        "uuid 231a9c9a-e5e6-58ff-97f2-3283dad269e0\nQuestion: what about bed two?")


@pytest.mark.parametrize("answer, flagged", [
    ("Patient ID 1234, bed label Bayside 3, has a critical alarm.", "patient id 1234"),
    ("Bed 1 and Bed 2", "bed 1"),
    ("Ward 4 Bay 1 (Bed 11) has the event.", "ward 4"),
    ("Ward 4 Bay 1 (Bed 11) has the event.", "bed 11"),
    ("Patient ID 657019a2-6542-505e-02a2e9cf3520.", "657019a2-6542-505e-02a2e9cf3520"),
    ("PT4543 is stable.", "PT4543"),
    ("Patient [P2] is in bed 1 and Patient [P3] is in bed 2.", "patient [P2]"),  # seen in replay
])
def test_invented_identifiers_are_flagged(answer, flagged):
    assert flagged in ungrounded_identifiers(answer, _CTX)


@pytest.mark.parametrize("answer", [
    "Patient PT992591 on bed Unit1-Bed03 had VT.",
    "Bed 3 had a ventricular tachycardia event.",                # Unit1-Bed03 is bed 3
    "Nothing is recorded for bed two.",                          # the clinician named it
    "Unit 1 has one event, 231a9c9a-e5e6-58ff-97f2-3283dad269e0.",
    "Escalate within 15 minutes and repeat the observations every 4 hours.",
])
def test_grounded_answers_pass(answer):
    assert ungrounded_identifiers(answer, _CTX) == []


# --- orchestrator wiring (A, C, D) -----------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answer


class _Working:
    def __init__(self):
        self.state = SimpleNamespace(patient_ref=None, event_ref=None, turns=[])

    def get_or_create(self, _sid):
        return self.state

    def save(self, _state):
        pass

    def append_turn(self, _sid, turn):
        self.state.turns.append(turn)


_SOP = Passage(text="Escalate a critical alarm within 15 minutes; call 212-555-0000.",
               source="critical_alarm_sop.md#Escalation", score=0.9)
_REPORT = Passage(text="Event report for Alice Smith, phone 212-555-0101.",
                  source="report:231a9c9a#Clinical Analysis", score=0.8)


class _Hybrid:
    def retrieve(self, _q, mode="hybrid"):
        return SimpleNamespace(passages=[_SOP, _REPORT], relationships=[
            Relationship(fact="Alice Smith HAS_CONDITION AF", source="graph:x")])


def _orch(answer):
    inner = _ScriptedLLM(answer)
    orch = Orchestrator(working=_Working(), hybrid=_Hybrid(),
                        episodic=SimpleNamespace(recall=lambda *a, **k: [], add=lambda *a, **k: None),
                        llm=DeidentifyingLLM(inner, RegexDeidentifier(names={"Alice Smith"})),
                        driver=None, episodic_recall=False, min_relevance=0.0)
    return orch, inner


def test_invented_patient_is_replaced_with_no_information(capsys):
    orch, _ = _orch("Patient ID 1234, bed label Bayside 3, has a critical alarm.")
    r = orch.handle_turn("s1", "when should I escalate a critical alarm?", now=NOW)
    assert r.answer == UNGROUNDED_ANSWER == "I don't have that patient information."
    assert "blocked answer naming identifier(s) not in context" in capsys.readouterr().out


def test_only_phi_capable_parts_are_deidentified():
    orch, inner = _orch("Escalate within 15 minutes.")
    orch.handle_turn("s1", "when should I escalate? Alice Smith 212-555-0199", now=NOW)
    sent = inner.prompts[-1]
    # SOP passage + citation markers reach the model intact ...
    assert "[P1] (critical_alarm_sop.md#Escalation)" in sent and "212-555-0000" in sent
    # ... while the question, patient report and graph facts are scrubbed.
    for phi in ("Alice Smith", "212-555-0199", "212-555-0101"):
        assert phi not in sent


def test_answer_echoing_a_placeholder_is_not_read_out():
    orch, _ = _orch("<NAME> is not associated with any unacknowledged alarms.")
    r = orch.handle_turn("s1", "when should I escalate a critical alarm?", now=NOW)
    assert "<NAME>" not in r.answer and "identifying details" in r.answer


@pytest.mark.parametrize("text, reply", [
    ("Thank you", "You're welcome."), ("Okay, thank you", "You're welcome."),
    ("thanks!", "You're welcome."), ("Goodbye", "Goodbye."), ("That's all, bye", "Goodbye."),
])
def test_courtesy_is_answered_without_the_kb(text, reply):
    orch, inner = _orch("should not be called")
    r = orch.handle_turn("s1", text, now=NOW)
    assert r.answer == reply and r.mode == "courtesy" and inner.prompts == []


# --- B. worklist questions ------------------------------------------------------------------------

@pytest.mark.parametrize("q, name", [
    ("How many events are there in my work list?", "worklist"),
    ("What are the events in my worklist?", "worklist"),
    ("Show the unacknowledged alarms", "worklist"),
    ("Which bed has the most severe alarm?", "worklist"),
    ("Which patient has the highest number of alarms?", "alarm_counts"),
    ("How many alarms per bed?", "alarm_counts"),
])
def test_worklist_questions_route_to_the_graph(q, name):
    assert match_intent(q, now=NOW)[0] == name


def test_worklist_hours_narrow_the_window():
    assert match_intent("unacknowledged alarms in the last 6 hours", now=NOW) == (
        "worklist", {"since": NOW - 6 * 3600})
    assert match_intent("what's on my worklist", now=NOW) == ("worklist", {"since": 0})


@pytest.mark.parametrize("q", [
    "What are the alarms in bed three?", "acknowledge the alarms pending with bin three",
    "what is the alarm on bed tree", "alarms on bed 3",
])
def test_spoken_bed_numbers_resolve_to_the_label(q):
    assert match_intent(q, now=NOW) == ("event_status_on_bed", {"bed": "Unit1-Bed03"})


def test_worklist_template_gets_its_defaults():
    seen = {}

    class _Driver:
        def run_read(self, cypher, **params):
            seen.update(params, cypher=cypher)
            return []

    run_template(_Driver(), "worklist")  # e.g. picked by the LLM router with no params
    assert seen["since"] == 0 and seen["levels"] == WORKLIST_LEVELS == ["High", "Critical"]
    assert seen["closed"] == CLOSED_STATUSES and "e.criticality IN $levels" in seen["cypher"]


def test_router_can_pick_the_worklist_templates():
    assert "worklist" in CATALOGUE and "alarm_counts" in CATALOGUE


def test_empty_worklist_says_so():
    assert _answer_operational([], "worklist").startswith("Your worklist is clear")
    assert _answer_operational([]) == "No matching records."


@pytest.mark.parametrize("text", ["ok", "okay", "great"])
def test_bare_acknowledgements_are_not_treated_as_thanks(text):
    # "ok"/"great" alone may be a reply mid-dialogue (e.g. to a question); only thanks/bye are
    # pleasantries. They fall through to the normal path.
    orch, _ = _orch("Escalate within 15 minutes.")
    assert orch.handle_turn("s1", text, now=NOW).mode != "courtesy"


# --- router: identities come from the clinician, documents stay documents ------------------------

class _RouterLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, _prompt, **_kw):
        self.calls += 1
        return self.reply


def test_router_cannot_invent_a_bed():
    from kb.graph.llm_router import route

    llm = _RouterLLM('{"name": "event_status_on_bed", "params": {"bed": "Unit1-Bed01"}}')
    assert route("what happened on the ward overnight?", llm, now=NOW) is None  # no bed named
    assert route("what happened on bed one overnight?", llm, now=NOW) == (
        "event_status_on_bed", {"bed": "Unit1-Bed01"})


def test_document_questions_skip_the_router():
    from kb.graph.llm_router import route

    llm = _RouterLLM('{"name": "protocol_for_bed_last_event", "params": {"bed": "Unit1-Bed01"}}')
    assert route("what is the protocol for treating VF?", llm, now=NOW) is None
    assert llm.calls == 0  # not even asked: no routing latency on document questions
