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
