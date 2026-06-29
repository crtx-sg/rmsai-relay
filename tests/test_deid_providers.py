"""De-identification + de-identifying LLM wrapper (PHI scrubbed before any model call)."""

from __future__ import annotations

import pytest

from common.deid import (
    DeidError,
    Deidentifier,
    RegexDeidentifier,
    _is_safe_span,
    deidentify,
    get_deidentifier,
)
from common.providers import DeidentifyingLLM, EchoLLM


# --- presidio allow-list: pseudonyms / clinical terms / bed-unit labels are NOT PHI ---


@pytest.mark.parametrize("span", ["PT4543", "patient PT99", "SVT", "ATRIAL_FIBRILLATION",
                                   "atrial fibrillation", "MEWS", "High", "Unit1", "Unit1-Bed01",
                                   "Bed01"])
def test_is_safe_span_true_for_non_phi(span):
    assert _is_safe_span(span) is True


@pytest.mark.parametrize("span", ["John Doe", "Jane Smith", "Springfield General", "Dr. House"])
def test_is_safe_span_false_for_real_phi(span):
    assert _is_safe_span(span) is False


def test_regex_keeps_dates_and_vitals_scrubs_phones():
    d = RegexDeidentifier()
    out = d.deidentify("At 2026-06-29 06:14 UTC, hr 68; sbp 118; spo2 97; call 212-555-0000")
    assert "2026-06-29" in out and "06:14" in out          # ISO date / time not a phone
    assert "68" in out and "118" in out and "97" in out      # vitals not phones
    assert "212-555-0000" not in out and "<PHONE>" in out    # a real 10-digit phone is scrubbed


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
