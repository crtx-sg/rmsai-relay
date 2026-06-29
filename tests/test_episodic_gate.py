"""Episodic recall is gated behind `episodic_recall` (off by default).

When off, free-text answers must not recall or archive past interactions, and the "Relevant past
interactions" block must not appear in the model input. Backends are faked so this stays offline.
"""

from __future__ import annotations

from types import SimpleNamespace

from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM, EchoLLM
from common.schemas import ChatTurn
from orchestrator.orchestrator import Orchestrator


class _FakeState:
    def __init__(self):
        self.patient_ref = None
        self.turns: list[ChatTurn] = []


class _FakeWorking:
    def __init__(self):
        self._s = _FakeState()

    def get_or_create(self, _sid):
        return self._s

    def save(self, _s):
        pass

    def append_turn(self, _sid, turn):
        self._s.turns.append(turn)


class _FakeHybrid:
    # one relationship -> the turn is "relevant" (not declined), so the LLM path runs.
    def retrieve(self, _q, mode="hybrid"):
        return SimpleNamespace(
            passages=[], relationships=[SimpleNamespace(source="graph:x", fact="afib relates to htn")]
        )


class _SpyEpisodic:
    def __init__(self):
        self.recalls = 0
        self.adds = 0

    def recall(self, *_a, **_k):
        self.recalls += 1
        return []

    def add(self, *_a, **_k):
        self.adds += 1


def _orch(ep, *, episodic_recall):
    return Orchestrator(
        working=_FakeWorking(), hybrid=_FakeHybrid(), episodic=ep,
        llm=DeidentifyingLLM(EchoLLM(), RegexDeidentifier()), driver=None,
        episodic_recall=episodic_recall,
    )


_Q = "explain rate control therapy"  # free-text -> hybrid LLM path (no operational intent)


def test_episodic_off_by_default_no_recall_no_block():
    ep = _SpyEpisodic()
    r = _orch(ep, episodic_recall=False).handle_turn("s", _Q)
    assert r.mode == "hybrid"
    assert ep.recalls == 0 and ep.adds == 0
    assert "## Relevant past interactions" not in r.model_input


def test_episodic_on_recalls_and_includes_block():
    ep = _SpyEpisodic()
    r = _orch(ep, episodic_recall=True).handle_turn("s", _Q)
    assert ep.recalls == 1 and ep.adds == 1
    assert "## Relevant past interactions" in r.model_input


# --- "repeat that" re-voices the last answer (conversational memory, no KB/LLM) ---


def test_repeat_revoices_last_answer():
    orch = _orch(_SpyEpisodic(), episodic_recall=False)
    first = orch.handle_turn("s", _Q)               # a normal answer
    for again in ["can you repeat that?", "say that again", "what did you just say"]:
        r = orch.handle_turn("s", again)
        assert r.mode == "repeat"
        assert r.answer == first.answer


def test_repeat_with_no_history():
    r = _orch(_SpyEpisodic(), episodic_recall=False).handle_turn("s", "repeat that")
    assert r.mode == "repeat"
    assert r.answer == "There is nothing to repeat yet."
