"""Guardrails — refuse-or-escalate around the orchestrator (Phase 8).

* **Input guardrail** (`check_input`) refuses unsafe requests before any retrieval/model call:
  prompt-injection / instruction-override, attempts to disclose secrets (the auth PIN), and
  requests to *take* a clinical action (the system informs, it does not act).
* **Output policy** (`decide_output`) turns a no-grounding situation into an **escalation** when the
  question signals a clinical emergency, instead of a bare decline.

Deterministic and rule-based for the POC; an LLM-judge guardrail can layer on later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|instruction|prompt|rule)s?\b",
    re.IGNORECASE,
)
_SECRET = re.compile(r"\b(pin|password|passcode|secret|api[\s_-]?key)\b", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(administer|give|inject|push|prescribe|set the dose|change the dose|increase the dose|"
    r"decrease the dose|defibrillate|shock|cardiovert|order)\b",
    re.IGNORECASE,
)
_EMERGENCY = re.compile(
    r"\b(emergency|arrest|code blue|coding|unresponsive|not breathing|dying|collapse|"
    r"seizure|anaphylaxis|stroke now)\b",
    re.IGNORECASE,
)

_REFUSE_INJECTION = "I can't change my safety instructions."
_REFUSE_SECRET = "I can't share authentication secrets."
_REFUSE_ACTION = (
    "I can't carry out clinical actions or orders — I can only provide information and guidance."
)
_ESCALATE = (
    "This sounds like an emergency. I can't safely answer that here — escalating to the on-call "
    "clinician now."
)


@dataclass
class InputDecision:
    allowed: bool
    message: str = ""  # refusal message when not allowed


# Patient pseudonyms, as common/deid.py preserves them (rule #6).
_PATIENT_REF = re.compile(r"\bPT\d+\b", re.IGNORECASE)

#: What the clinician hears instead of an answer that names a patient the model was never given.
UNCONFIRMED_PATIENT = ("I can't confirm that for this patient from the information I have. "
                       "Please check the event details.")


def foreign_patient_refs(answer: str, context: str, session_patient: str | None = None) -> list[str]:
    """Patient ids named in a model answer that appear neither in the context it was given nor as
    the session's patient. Any hit means the model invented or misattributed a patient, which on a
    clinical relay is worse than no answer, so the caller replaces the answer."""
    allowed = {m.upper() for m in _PATIENT_REF.findall(context)}
    if session_patient:
        allowed.add(session_patient.upper())
    return sorted({m.upper() for m in _PATIENT_REF.findall(answer)} - allowed)


#: What the clinician hears instead of an answer that names a patient, bed, ward or event id that is
#: neither in the context the model was given nor in the clinician's own words.
UNGROUNDED_ANSWER = "I don't have that patient information."

# Identifiers a small model invents when it has no patient data ("Patient ID 1234", "Bed 11",
# "Ward 4 Bay 1", a mangled event uuid). Each must be traceable to the prompt (context, history or
# the question itself), or the answer is replaced.
_PATIENT_ID = re.compile(r"\bpatient\s*(?:id|number|no\.?|#)\s*[:#]?\s*([A-Za-z0-9][\w-]*)",
                         re.IGNORECASE)
_BED_LABEL = re.compile(r"\b[A-Za-z]+\d*-Bed(\d+)\b", re.IGNORECASE)
_BED_WORD = re.compile(r"\bbeds?\s+(?:no\.?\s*|number\s+|#\s*)?(\d+|[a-z]+)\b", re.IGNORECASE)
_PLACE = re.compile(r"\b(ward|bay|room|unit)\s*(\d+|[A-Z]\b)", re.IGNORECASE)
_HEX_ID = re.compile(r"\b[0-9a-f]{6,}(?:-[0-9a-f]{3,})+\b", re.IGNORECASE)
# A citation marker used as a person ("Patient [P2] is in bed 1"): passages are not patients.
_CITED_AS_PATIENT = re.compile(r"\bpatients?\s*(?:id\s*)?[:#]?\s*\[([PR]\d+)\]", re.IGNORECASE)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _bed_numbers(text: str) -> set[int]:
    nums = {int(n) for n in _BED_LABEL.findall(text)}
    for tok in _BED_WORD.findall(text):
        tok = tok.lower()
        if tok.isdigit():
            nums.add(int(tok))
        elif tok in _NUMBER_WORDS:
            nums.add(_NUMBER_WORDS[tok])
    return nums


def ungrounded_identifiers(answer: str, context: str, session_patient: str | None = None) -> list[str]:
    """Identifiers in a model answer that the model was never given. Empty list = grounded.

    `context` is everything the model saw (instructions, history, retrieved blocks, the question), so
    a bed or patient the clinician named is allowed. Checks patient pseudonyms (`PT…`), "patient ID
    <x>", bed labels/numbers, ward/bay/room/unit numbers, and uuid-like event ids.
    """
    found = set(foreign_patient_refs(answer, context, session_patient))
    ctx = context.lower()
    ctx_compact = re.sub(r"\s+", "", ctx)
    for pid in _PATIENT_ID.findall(answer):
        if pid.lower() not in ctx and pid.upper() != (session_patient or "").upper():
            found.add(f"patient id {pid}")
    allowed_beds = _bed_numbers(context)
    for n in sorted(_bed_numbers(answer) - allowed_beds):
        found.add(f"bed {n}")
    for kind, num in _PLACE.findall(answer):
        if f"{kind}{num}".lower() not in ctx_compact:
            found.add(f"{kind.lower()} {num}")
    for marker in _CITED_AS_PATIENT.findall(answer):
        found.add(f"patient [{marker}]")
    for hex_id in _HEX_ID.findall(answer):
        if hex_id.lower() not in ctx:
            found.add(hex_id)
    return sorted(found)


def check_input(text: str) -> InputDecision:
    """Refuse unsafe inputs before retrieval/model. Returns allowed=False + a refusal message."""
    if _INJECTION.search(text):
        return InputDecision(False, _REFUSE_INJECTION)
    if _SECRET.search(text) and re.search(r"\b(what|tell|give|share|reveal)\b", text, re.IGNORECASE):
        return InputDecision(False, _REFUSE_SECRET)
    if _ACTION.search(text):
        return InputDecision(False, _REFUSE_ACTION)
    return InputDecision(True)


def decide_output(user_text: str, *, declined: bool) -> str:
    """Map (declined?) + content to an action: 'answer' | 'escalate' | 'decline'."""
    if declined and _EMERGENCY.search(user_text):
        return "escalate"
    if declined:
        return "decline"
    return "answer"


class Guardrails:
    """Bundles the input + output guardrails (injectable so tests/policies can vary)."""

    refusal_for_emergency = _ESCALATE

    def check_input(self, text: str) -> InputDecision:
        return check_input(text)

    def decide_output(self, user_text: str, *, declined: bool) -> str:
        return decide_output(user_text, declined=declined)
