"""De-identification — PHI scrubbing before any model call (fail closed).

Data in the pipeline is pseudonymized by construction (patients are `PT####`), so this is a
defence-in-depth net that catches accidental PHI in free text. `RegexDeidentifier` is the
offline default (emails, phones, SSNs, MRNs, and a registered name set); `PresidioDeidentifier`
is the real NLP-backed backend (lazy import). `deidentify` **fails closed**: if the backend
errors, it raises rather than letting un-scrubbed text through.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b\+?\d[\d\-().\s]{7,}\d\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN = re.compile(r"\bMRN[:#]?\s*\d+\b", re.IGNORECASE)


class DeidError(Exception):
    """Raised when de-identification fails — callers must NOT send text to a model."""


class Deidentifier(ABC):
    @abstractmethod
    def deidentify(self, text: str) -> str: ...


class RegexDeidentifier(Deidentifier):
    """Pattern-based scrubber + an explicit name registry (no NLP needed)."""

    def __init__(self, names: set[str] | None = None) -> None:
        self._names = {n for n in (names or set()) if n}

    def add_name(self, name: str) -> None:
        if name:
            self._names.add(name)

    def deidentify(self, text: str) -> str:
        text = _EMAIL.sub("<EMAIL>", text)
        text = _SSN.sub("<SSN>", text)
        text = _MRN.sub("<MRN>", text)
        text = _PHONE.sub("<PHONE>", text)
        for name in self._names:
            text = re.sub(rf"\b{re.escape(name)}\b", "<NAME>", text, flags=re.IGNORECASE)
        return text


class PresidioDeidentifier(Deidentifier):
    """NLP-backed de-identification via Microsoft Presidio (lazy import)."""

    def __init__(self, language: str = "en") -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._language = language

    def deidentify(self, text: str) -> str:
        results = self._analyzer.analyze(text=text, language=self._language)
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text


def deidentify(deidentifier: Deidentifier, text: str) -> str:
    """De-identify text, failing closed on any backend error."""
    try:
        return deidentifier.deidentify(text)
    except Exception as exc:  # noqa: BLE001
        raise DeidError(f"de-identification failed: {exc}") from exc


def get_deidentifier(name: str = "auto", names: set[str] | None = None) -> Deidentifier:
    """'auto' tries Presidio then falls back to regex; 'regex'/'presidio' force one."""
    if name in ("presidio", "auto"):
        try:
            return PresidioDeidentifier()
        except Exception:  # noqa: BLE001 - presidio/spacy model unavailable
            if name == "presidio":
                raise
    return RegexDeidentifier(names)
