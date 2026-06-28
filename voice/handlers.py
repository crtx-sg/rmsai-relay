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
from voice.outbound import parse_ack
from voice.outbound_alert import OutboundAlert


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

    def is_authenticated(self, session_id: str) -> bool:
        """Whether this session passed the PIN gate — i.e. is in the post-alert Q&A phase.

        The LiveKit worker uses this to scope wake-word gating to follow-up audio only (PIN entry,
        the spoken alert, and the verbal ack run before this is True and are never gated).
        """
        return self.working.get_or_create(session_id).authenticated

    def respond(self, text: str, *, session_id: str) -> str:
        state = self.working.get_or_create(session_id)

        if state.authenticated:
            return self._authenticated_turn(session_id, text)

        # --- not yet authenticated: PIN gate ---
        if self.auth_gate.verify(text):
            self.working.set_authenticated(session_id)
            self.audit.write(actor=f"caller:{session_id}", action="inbound_auth",
                             subject=session_id, outcome="success")
            print(f"[voice] PIN accepted for session {session_id}", flush=True)
            return self._on_authenticated(session_id)

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

    def _on_authenticated(self, session_id: str) -> str:
        """First message spoken right after the PIN is accepted. Overridable (see OutboundHandler)."""
        return _AUTH_OK

    def _authenticated_turn(self, session_id: str, text: str) -> str:
        """Handle one post-auth turn. Overridable; default routes to the grounded orchestrator."""
        result = self.orchestrator.handle_turn(session_id, text)
        self.audit.write(
            actor=f"caller:{session_id}", action="phi_voice_query",
            subject=self.working.get_or_create(session_id).patient_ref or session_id,
            outcome="declined" if result.declined else "answered",
        )
        return result.answer


_ACK_CONFIRMED = "Thank you, I've recorded your acknowledgment. Goodbye."


class OutboundHandler(OrchestratorHandler):
    """Outbound (relay-initiated) variant: PIN gate, then voice *this event's* alert, then Q&A + ack.

    The relay placed the call for a specific `OutboundAlert`; this handler is seeded with it. The PIN
    gate is unchanged (PHI stays fail-closed). On auth it binds the session to the alert's patient so
    follow-ups are scoped, and speaks the event report instead of a generic greeting. A spoken
    acknowledgment flips the `MonitoredEvent` status in the graph (G4) — closing the outbound loop
    from the worker side.
    """

    def __init__(self, orchestrator, working_memory, alert: OutboundAlert, *, driver=None, **kwargs) -> None:
        super().__init__(orchestrator, working_memory, **kwargs)
        self.alert = alert
        self.driver = driver

    def _on_authenticated(self, session_id: str) -> str:
        # Scope every subsequent turn to this patient, then speak the alert that prompted the call.
        self.working.set_authenticated(session_id, patient_ref=self.alert.patient_ref)
        print(f"[voice] speaking event alert for {self.alert.patient_ref} "
              f"(event {self.alert.event_id})", flush=True)
        return self.alert.spoken_alert

    def _authenticated_turn(self, session_id: str, text: str) -> str:
        if parse_ack(text) == "yes":
            if self.driver is not None:
                from kb.graph.events import set_event_status  # noqa: PLC0415

                set_event_status(self.driver, self.alert.event_id, "acknowledged")
            self.audit.write(actor=f"caller:{session_id}", action="acknowledgment",
                             subject=self.alert.patient_ref, outcome="acknowledged")
            print(f"[voice] acknowledgment received -> MonitoredEvent {self.alert.event_id} "
                  f"status=acknowledged (Neo4j updated)", flush=True)
            return _ACK_CONFIRMED
        return super()._authenticated_turn(session_id, text)


def build_handler(
    mode: str = "orchestrator", *, embedder: str = "hashing", llm: str = "echo",
) -> tuple[Handler, str | None, "callable | None"]:
    """Build a conversation handler for `mode`. Returns (handler, greeting, cleanup).

    Single source of truth shared by the offline `cli.voice` demo and the live LiveKit worker.
    `greeting` is the line to speak first (None for echo); `cleanup` releases backend resources
    (None for echo). `embedder`/`llm` keep the terminal demo offline by default; the de-id backend
    still honours config (DEID_BACKEND).
    """
    if mode == "echo":
        return EchoHandler(), None, None

    from orchestrator.chat import build_orchestrator  # noqa: PLC0415 - heavy, optional dep

    orch, driver = build_orchestrator(embedder=embedder, llm=llm)
    handler: Handler = OrchestratorHandler(orch, orch.working)
    return handler, handler.greeting(), driver.close


def build_outbound_handler(
    alert: OutboundAlert, *, embedder: str = "hashing", llm: str = "echo",
) -> tuple[Handler, str | None, "callable | None"]:
    """Build the outbound (event-seeded) handler for the worker. Returns (handler, greeting, cleanup).

    Same backends as `build_handler`, but wrapped in an `OutboundHandler` carrying the alert + a
    graph driver (so a spoken acknowledgment updates the event status). Greeting is the PIN prompt.
    """
    from orchestrator.chat import build_orchestrator  # noqa: PLC0415

    orch, driver = build_orchestrator(embedder=embedder, llm=llm)
    handler = OutboundHandler(orch, orch.working, alert, driver=driver)
    return handler, handler.greeting(), driver.close
