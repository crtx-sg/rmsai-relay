"""Phase 8 failure modes: LLM timeout/fail-closed generation (unit, no backends)."""

from __future__ import annotations

from common.deid import DeidError
from common.interfaces import LLMProvider
from orchestrator.orchestrator import _LLM_FALLBACK, Orchestrator


class _TimeoutLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        raise TimeoutError("model timed out")

    def embed(self, texts):
        return [[0.0] for _ in texts]


class _DeidFailLLM(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        raise DeidError("de-id failed")

    def embed(self, texts):
        return [[0.0] for _ in texts]


def _orch(llm, retries=1) -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)  # bypass backend wiring; _generate only needs llm
    o.llm = llm
    o.llm_retries = retries
    return o


def test_llm_timeout_retries_then_falls_back():
    llm = _TimeoutLLM()
    text, failed = _orch(llm, retries=1)._generate("prompt")
    assert failed and text == _LLM_FALLBACK
    assert llm.calls == 2  # initial + 1 retry, then fallback (never crashes)


def test_deid_failure_is_fail_closed():
    text, failed = _orch(_DeidFailLLM())._generate("prompt with PHI")
    assert failed and "couldn't safely process" in text  # no model output produced
