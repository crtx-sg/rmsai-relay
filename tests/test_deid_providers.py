"""De-identification + de-identifying LLM wrapper (PHI scrubbed before any model call)."""

from __future__ import annotations

import pytest

from common.deid import DeidError, Deidentifier, RegexDeidentifier, deidentify, get_deidentifier
from common.providers import DeidentifyingLLM, EchoLLM


# --- regex de-id ---


def test_scrubs_structured_phi():
    d = RegexDeidentifier()
    out = d.deidentify("call John at 415-555-1234 or jane@hosp.org, MRN: 0099887, SSN 123-45-6789")
    assert "415-555-1234" not in out and "<PHONE>" in out
    assert "jane@hosp.org" not in out and "<EMAIL>" in out
    assert "0099887" not in out and "<MRN>" in out
    assert "123-45-6789" not in out and "<SSN>" in out


def test_scrubs_registered_names():
    d = RegexDeidentifier(names={"John Doe"})
    assert "John Doe" not in d.deidentify("patient John Doe presented with chest pain")


def test_pseudonym_passes_through():
    d = RegexDeidentifier()
    assert "PT1234" in d.deidentify("event for PT1234 on bed Unit1-Bed03")


# --- fail closed ---


class _Broken(Deidentifier):
    def deidentify(self, text: str) -> str:
        raise RuntimeError("boom")


def test_deidentify_fails_closed():
    with pytest.raises(DeidError):
        deidentify(_Broken(), "anything")


def test_get_deidentifier_falls_back_to_regex():
    assert isinstance(get_deidentifier("regex"), RegexDeidentifier)
    assert isinstance(get_deidentifier("auto"), Deidentifier)


# --- de-identifying LLM wrapper ---


def test_wrapper_scrubs_before_model_sees_it():
    inner = EchoLLM()
    llm = DeidentifyingLLM(inner, RegexDeidentifier(names={"Alice Smith"}))
    llm.generate("Question: how is Alice Smith (phone 212-555-0000) doing?")
    seen = inner.last_prompt
    assert "Alice Smith" not in seen and "212-555-0000" not in seen
    assert "<NAME>" in seen and "<PHONE>" in seen


def test_wrapper_fails_closed_blocks_model_call():
    inner = EchoLLM()
    llm = DeidentifyingLLM(inner, _Broken())
    with pytest.raises(DeidError):
        llm.generate("anything with PHI")
    assert inner.prompts == []  # model never called
