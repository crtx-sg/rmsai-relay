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

from .config import DEFAULT
from .event_types import CLASS_NAMES

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone *candidate*; only scrubbed when it actually has >=10 digits, so date/time/vital numbers
# ("2026-06-29", "06:14", "118") aren't mistaken for phone numbers.
_PHONE = re.compile(r"\b\+?\d[\d\-().\s]{7,}\d\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN = re.compile(r"\bMRN[:#]?\s*\d+\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\d{4}-\d\d-\d\d")  # a phone candidate containing this is really a date


def _phone_sub(m: re.Match) -> str:
    """Scrub a phone candidate only if it's a real phone: >=10 digits and not an ISO date."""
    s = m.group()
    if _ISO_DATE.search(s) or sum(c.isdigit() for c in s) < 10:
        return s
    return "<PHONE>"
# Patient pseudonym ("PT4543") — the de-identified reference the whole system uses (rule #6). It is
# NOT PHI, so de-id must preserve it; presidio's NER otherwise flags it as PERSON/ORGANIZATION.
_PSEUDONYM = re.compile(r"\bPT\d+\b", re.IGNORECASE)

# Clinical vocabulary presidio's NER mis-flags as PERSON/ORGANIZATION (e.g. "SVT" -> ORGANIZATION),
# which would gut operational answers. These terms are not PHI, so de-id preserves them. Sourced
# from the 16 event-type classes (+ spaced forms) plus the acronyms/labels that show up in answers.
_CLINICAL_TERMS = (
    {c.lower() for c in CLASS_NAMES}
    | {c.replace("_", " ").lower() for c in CLASS_NAMES}
    | {"svt", "afib", "a-fib", "vt", "v-tach", "vf", "vfib", "pvc", "pac", "lbbb", "rbbb", "stemi",
       "mews", "spo2", "hr", "bp", "rr", "sbp", "dbp", "ecg", "stemi", "bradycardia", "tachycardia",
       "fibrillation", "flutter", "high", "medium", "low", "critical", "normal", "sinus"}
)
_CLINICAL_RE = re.compile(
    r"^(?:" + "|".join(re.escape(t) for t in sorted(_CLINICAL_TERMS, key=len, reverse=True)) + r")$",
    re.IGNORECASE,
)


# Bed / unit labels ("Unit1", "Unit1-Bed01", "Bed01") — operational identifiers, not PHI; presidio
# otherwise flags them as ORGANIZATION.
_BED_UNIT = re.compile(r"\b(?:unit|ward|bed)\s*[\w-]*\d[\w-]*\b", re.IGNORECASE)


def _is_safe_span(span: str) -> bool:
    """A de-id-flagged span that is really a pseudonym / clinical term / bed-unit label (not PHI)."""
    s = span.strip()
    return bool(_PSEUDONYM.search(s) or _CLINICAL_RE.match(s) or _BED_UNIT.search(s))


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
        text = _PHONE.sub(_phone_sub, text)  # only a real phone, not a date/time/vital number
        for name in self._names:
            text = re.sub(rf"\b{re.escape(name)}\b", "<NAME>", text, flags=re.IGNORECASE)
        return text


# Direct identifiers to scrub. Deliberately omits presidio's quasi-identifier entities — DATE_TIME,
# ORGANIZATION, LOCATION, NRP, CARDINAL/MONEY/PERCENT — which here only mangle operational text
# ("2026-06-29"->DATE_TIME, "UTC"->ORGANIZATION, the "2" of "S P O 2"->DATE_TIME) without protecting
# real PHI (patients are pseudonymized by construction; cloud paths are synthetic-only).
_PRESIDIO_ENTITIES = [
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "US_ITIN", "US_DRIVER_LICENSE",
    "US_PASSPORT", "US_BANK_NUMBER", "CREDIT_CARD", "IBAN_CODE", "CRYPTO", "IP_ADDRESS",
    "MEDICAL_LICENSE", "URL",
]


class PresidioDeidentifier(Deidentifier):
    """NLP-backed de-identification via Microsoft Presidio (lazy import).

    `spacy_model` selects the spaCy NER model (default `en_core_web_lg`; use `en_core_web_sm` for
    a lighter install). The model must be downloaded first:
    `uv run python -m spacy download <model>`.
    """

    def __init__(self, language: str = "en", spacy_model: str | None = None) -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: PLC0415
        from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415

        spacy_model = spacy_model or DEFAULT.deid_spacy_model
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
        }).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
        self._anonymizer = AnonymizerEngine()
        self._language = language

    def deidentify(self, text: str) -> str:
        results = self._analyzer.analyze(
            text=text, language=self._language, entities=_PRESIDIO_ENTITIES
        )
        # Belt-and-suspenders: drop any span that is actually a pseudonym / clinical term / bed-unit
        # label (e.g. a name-like pseudonym flagged PERSON). Real PHI (names, phones, …) is untouched.
        results = [r for r in results if not _is_safe_span(text[r.start:r.end])]
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
