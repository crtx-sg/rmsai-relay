"""Conversation handlers — text in, text out.

`EchoHandler` parrots the caller (Phase 5, to prove the audio loop). `OrchestratorHandler`
(Phase 6) authenticates the caller with a shared PIN **before any PHI is voiced**, then routes to
the Phase 4 text orchestrator for grounded answers. Both satisfy `Handler`, so the voice session
is unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.audit import AuditLog
from voice.auth import PinAuthGate, parse_pin


class Handler(ABC):
    @abstractmethod
    def respond(self, text: str, *, session_id: str) -> str: ...


class EchoHandler(Handler):
    """Returns what it heard (a parrot), proving STT -> handler -> TTS works end-to-end."""

    def respond(self, text: str, *, session_id: str) -> str:
        return text


_PROMPT_PIN = "Please say or enter your four digit PIN to continue."
_AUTH_OK = "Thank you, you are authenticated. How can I help?"
_LOCKED = "I could not verify your PIN. Ending the call for safety."


class OrchestratorHandler(Handler):
    """PHI-gated voice handler: shared-PIN auth, then grounded answers from the orchestrator.

    Until the session is authenticated, this NEVER calls the orchestrator, so no PHI can be voiced
    (fail closed). The orchestrator's de-identifying LLM is a second layer behind that.
    """

    def __init__(
        self,
        orchestrator,
        working_memory,
        *,
        auth_gate: PinAuthGate | None = None,
        audit: AuditLog | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.orchestrator = orchestrator
        self.working = working_memory
        self.auth_gate = auth_gate or PinAuthGate()
        self.audit = audit or AuditLog()
        self.max_attempts = max_attempts
        self._attempts: dict[str, int] = {}

    def greeting(self) -> str:
        return f"Remote clinical line. {_PROMPT_PIN}"

    def respond(self, text: str, *, session_id: str) -> str:
        state = self.working.get_or_create(session_id)

        if state.authenticated:
            result = self.orchestrator.handle_turn(session_id, text)
            self.audit.write(
                actor=f"caller:{session_id}", action="phi_voice_query",
                subject=state.patient_ref or session_id,
                outcome="declined" if result.declined else "answered",
            )
            return result.answer

        # --- not yet authenticated: PIN gate ---
        if self.auth_gate.verify(text):
            self.working.set_authenticated(session_id)
            self.audit.write(actor=f"caller:{session_id}", action="inbound_auth",
                             subject=session_id, outcome="success")
            return _AUTH_OK

        if parse_pin(text):  # looked like a PIN but was wrong -> count an attempt
            self._attempts[session_id] = self._attempts.get(session_id, 0) + 1
            self.audit.write(actor=f"caller:{session_id}", action="inbound_auth",
                             subject=session_id, outcome="failure",
                             attempt=self._attempts[session_id])
            if self._attempts[session_id] >= self.max_attempts:
                return _LOCKED
            return f"That PIN was not recognised. {_PROMPT_PIN}"

        # not a PIN at all -> refuse PHI, prompt (no attempt charged)
        return f"I can't share patient information until you authenticate. {_PROMPT_PIN}"
