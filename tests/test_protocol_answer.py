"""Protocol questions answered from structured steps, else the document's sections — no LLM."""

from __future__ import annotations

from types import SimpleNamespace

from common.schemas import ChatTurn
from kb.vector.store import SearchHit
from orchestrator import protocol_answer as pa
from orchestrator.orchestrator import Orchestrator

_AFIB = {"protocol": "Atrial fibrillation with rapid ventricular response", "source": "x", "steps": [
    "1. [vitals_check] vitals_check (vital=HR, periodicity=q15min x4 then hourly, alert_if=HR > 150)",
    "2. [test] test (name=12-lead ECG, when=on detection, priority=high)",
    "3. [medication] medication (drug=rate-control per local policy, route=IV, caution=check BP)",
    "4. [escalation] escalation (trigger=HR > 150 or SBP < 90, escalate_to=on-call cardiology, "
    "within=15 min)",
    "5. [monitoring] monitoring (modality=continuous ECG, duration=24h)"]}

_PVC = "p.md#3. Premature Ventricular Complexes (PVCs) Protocol"
_HITS = [
    SearchHit(text="3. PVCs › Long-Term Management\n*   **First-Line:** Oral beta-blockers.\n"
              "*   **Ablation:** For refractory PVCs", source=f"{_PVC} › Long-Term Management",
              doc_id="p.md", score=0.9),
    SearchHit(text="2. SVT › Acute\n*   **First Line:** Vagal maneuvers.",  # not PVC: never read
              source="p.md#2. Supraventricular Tachycardia (SVT) Protocol › Acute", doc_id="p.md",
              score=0.95),
    SearchHit(text="3. PVCs › Acute\n*   **Asymptomatic:** No acute treatment.\n"
              "*   **Symptomatic:**\n    *   **Electrolytes:** Keep K+ >4.0.",
              source=f"{_PVC} › Acute & Emergency Care", doc_id="p.md", score=0.8),
    SearchHit(text="3. PVCs › Long-Term\n*Disclaimer: informational only.*",
              source=f"{_PVC} › Long-Term Management", doc_id="p.md", score=0.7),
    SearchHit(text="Classification\nPVC 0.65", source="report:abc#Classification", doc_id="report:abc",
              score=0.99),
    SearchHit(text="3. PVCs Long-Term Care\n*   Holter.", source="other.md#3. PVCs Long-Term Care",
              doc_id="other.md", score=0.85),
]


def test_step_is_spoken_as_plain_words():
    assert pa.speak_step(_AFIB["steps"][3]) == (
        "Escalation: if HR > 150 or SBP < 90, escalate to on-call cardiology, within 15 min")
    assert pa.speak_step(_AFIB["steps"][0]) == "Vitals check: HR, q15min x4 then hourly, alert if HR > 150"


def test_spoken_steps_stop_at_three_and_go_on_keeps_numbers():
    first = pa.answer_from_steps(_AFIB, "ATRIAL_FIBRILLATION", spoken=True, question="AFib protocol")
    assert "3. Medication" in first.text and "4." not in first.text
    assert first.text.endswith("2 more steps; ask for the full protocol.")
    rest = pa.answer_from_steps(_AFIB, "ATRIAL_FIBRILLATION", spoken=True, question="go on")
    assert "4. Escalation" in rest.text and "5. Monitoring" in rest.text and "1." not in rest.text
    full = pa.answer_from_steps(_AFIB, "ATRIAL_FIBRILLATION", spoken=True, question="the full one")
    assert "1. Vitals" in full.text and "5. Monitoring" in full.text


def test_sections_come_from_one_document_in_heading_order():
    a = pa.answer_from_sections(_HITS, "PVC", spoken=True, question="protocol for PVC")
    # SVT section, report and disclaimer are left out; acute (heading order) before long-term.
    assert a.text == ("PVC protocol. Acute and Emergency Care: Asymptomatic: No acute treatment. "
                      "Symptomatic: Electrolytes: Keep K+ >4.0. Long-Term Management: First-Line: "
                      "Oral beta-blockers. 1 more point, on Long-Term Management; ask for the full "
                      "PVC protocol, or for one of those sections.")
    assert "other.md" not in " ".join(a.citations) and all("SVT" not in c for c in a.citations)
    rest = pa.answer_from_sections(_HITS, "PVC", spoken=True, question="tell me the rest")
    assert rest.text == "PVC protocol. Long-Term Management: Ablation: For refractory PVCs."


def test_a_section_can_be_asked_for_by_name():
    a = pa.answer_from_sections(_HITS, "PVC", spoken=True, question="its long term management?")
    assert "Acute" not in a.text and "First-Line: Oral beta-blockers." in a.text


def test_no_section_for_the_rhythm_is_no_answer():
    assert pa.answer_from_sections(_HITS, "LBBB", spoken=True, question="LBBB protocol") is None


# --- orchestrator: no LLM call, follow-ups, other questions untouched ----------------------------

class _NoLLM:
    def generate(self, *_a, **_k):
        raise AssertionError("protocol answers must not call the model")


class _Working:
    def __init__(self):
        self.state = SimpleNamespace(patient_ref=None, event_ref=None, turns=[])

    def get_or_create(self, _sid):
        return self.state

    def save(self, _state):
        pass

    def append_turn(self, _sid, turn: ChatTurn):
        self.state.turns.append(turn)


def _orch():
    vector = SimpleNamespace(search=lambda *_a, **_k: _HITS)
    return Orchestrator(working=_Working(), hybrid=SimpleNamespace(vector=vector), llm=_NoLLM(),
                        episodic=SimpleNamespace(recall=lambda *a, **k: [], add=lambda *a, **k: None),
                        driver=None, episodic_recall=False, min_relevance=0.0)


def test_protocol_question_and_its_follow_up_skip_the_model():
    orch = _orch()
    r = orch.handle_turn("s", "What is the protocol to handle PVC?", now=1.0, spoken=True)
    assert r.mode == "kb" and r.answer.startswith("PVC protocol. Acute and Emergency Care")
    r = orch.handle_turn("s", "what about its long term management?", now=2.0, spoken=True)
    assert r.answer.startswith("PVC protocol. Long-Term Management")


def test_rhythm_without_protocol_words_is_not_hijacked():
    assert _orch()._protocol_answer("how many PVC events today?", _Working().state, True) is None


def test_rvr_steps_only_for_rvr_questions():
    assert pa.steps_apply(_AFIB, "how do you manage AFib with RVR?")
    assert pa.steps_apply(_AFIB, "protocol for atrial fibrillation with rapid ventricular response")
    assert not pa.steps_apply(_AFIB, "what is the protocol for atrial fibrillation?")
    assert pa.steps_apply({"protocol": "Default monitoring protocol"}, "protocol for AFib")


def test_afib_without_rvr_reads_the_documents_and_go_on_keeps_rvr():
    afib = SearchHit(text="1. AFib › Acute\n*   **Unstable:** Synchronized cardioversion.",
                     source="p.md#1. Atrial Fibrillation (AFib) Protocol › Acute", doc_id="p.md",
                     score=0.9)
    orch = _orch()
    orch.hybrid.vector.search = lambda *_a, **_k: [afib]
    orch.driver = object()
    import orchestrator.orchestrator as om
    real = om.run_template
    om.run_template = lambda *_a, **_k: [_AFIB]
    try:
        r = orch.handle_turn("s", "what is the protocol for AFib?", now=1.0, spoken=True)
        assert r.answer.startswith("Atrial fibrillation protocol. Acute: Unstable")
        r = orch.handle_turn("s", "how do you manage AFib with RVR?", now=2.0, spoken=True)
        assert "3. Medication" in r.answer and "4." not in r.answer
        r = orch.handle_turn("s", "go on", now=3.0, spoken=True)
        assert "4. Escalation" in r.answer
    finally:
        om.run_template = real
