"""Text orchestrator — the stateful chat state machine (Phase 4).

Per turn: load state (working memory) -> classify intent -> retrieve (operational template OR
hybrid KB) + recall episodic -> build a labelled, de-identified context -> LLM -> respond ->
persist (working + episodic). The LLM is always a `DeidentifyingLLM`, so PHI is scrubbed before
any model call by construction.

This is a hand-rolled node pipeline; LangGraph can wrap these same nodes later (the working-memory
checkpointer and node boundaries are already in place).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from common.providers import DeidentifyingLLM
from common.schemas import ChatTurn
from common.tracing import Tracer
from kb.graph.driver import GraphDriver
from kb.graph.lookup import match_intent
from kb.graph.templates import run_template
from kb.hybrid.retriever import HybridRetriever
from kb.vector.answer import _best_overlap
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory

from .guardrails import Guardrails

_MIN_OVERLAP = 0.18
_DECLINE = "I don't have information on that in the knowledge base."
_LLM_FALLBACK = "I'm having trouble generating a response right now; please try again."
_MAX_HISTORY = 6


@dataclass
class TurnResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    mode: str = "hybrid"
    declined: bool = False
    refused: bool = False
    escalated: bool = False
    model_input: str = ""  # the (de-identified) text actually sent to the model — for audit/tests
    trace: list[dict] = field(default_factory=list)


def _render_blocks(result) -> tuple[str, list[str]]:
    passages = "\n\n".join(f"[P{i + 1}] ({p.source}) {p.text}" for i, p in enumerate(result.passages))
    rels = "\n".join(f"[R{i + 1}] ({r.source}) {r.fact}" for i, r in enumerate(result.relationships))
    text = f"## Retrieved passages\n{passages or '(none)'}\n\n## Known relationships\n{rels or '(none)'}"
    cites = [p.source for p in result.passages] + [r.source for r in result.relationships]
    return text, cites


_TS_KEYS = {"ts", "timestamp", "due", "due_at", "generated_at"}


def _fmt_value(key: str, value) -> str:
    """Human-readable scalar: trim float noise, render epoch timestamps as a date-time."""
    if key in _TS_KEYS and isinstance(value, (int, float)):
        from datetime import datetime, timezone  # noqa: PLC0415

        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, float):
        return f"{value:g}"  # 171.0 -> "171", 97.9 -> "97.9"
    return str(value)


def _render_operational(name: str, rows: list[dict]) -> str:
    """Render template rows as readable text (not raw dicts), so the answer reads as prose.

    Each row becomes `- key: value, key: value` with nulls dropped and floats/timestamps tidied.
    Feeding the LLM clean text (instead of `{'hr': 171.0, ...}`) keeps the grounded answer from
    echoing a Python object back to the clinician.
    """
    if not rows:
        return f"## Operational result: {name}\n(no matching records)"
    lines = []
    for row in rows:
        pairs = [f"{k}: {_fmt_value(k, v)}" for k, v in row.items() if v is not None]
        lines.append(f"- {', '.join(pairs)}")
    return f"## Operational result: {name}\n" + "\n".join(lines)


class Orchestrator:
    def __init__(
        self,
        working: WorkingMemory,
        hybrid: HybridRetriever,
        episodic: EpisodicMemory,
        llm: DeidentifyingLLM,
        driver: GraphDriver,
        *,
        guardrails: Guardrails | None = None,
        llm_retries: int = 1,
    ) -> None:
        self.working = working
        self.hybrid = hybrid
        self.episodic = episodic
        self.llm = llm
        self.driver = driver
        self.guardrails = guardrails or Guardrails()
        self.llm_retries = llm_retries

    def _generate(self, prompt: str) -> tuple[str, bool]:
        """Generate with one retry on transient failure; return (text, failed). Never raises."""
        from common.deid import DeidError  # noqa: PLC0415

        for _ in range(self.llm_retries + 1):
            try:
                return self.llm.generate(prompt), False
            except DeidError:
                return "I couldn't safely process that request.", True  # fail closed
            except Exception:  # noqa: BLE001 - timeout/rate-limit/unreachable -> retry then fall back
                continue
        return _LLM_FALLBACK, True

    def handle_turn(
        self, session_id: str, user_text: str, *, patient_ref: str | None = None, now: float | None = None
    ) -> TurnResult:
        now = now if now is not None else time.time()
        tracer = Tracer()

        with tracer.span("load_state"):
            state = self.working.get_or_create(session_id)
            if patient_ref:
                state.patient_ref = patient_ref
                self.working.save(state)
            self.working.append_turn(
                session_id, ChatTurn(role="clinician", text=user_text, timestamp=now)
            )

        # --- input guardrail: refuse unsafe requests before any retrieval / model call ---
        decision = self.guardrails.check_input(user_text)
        if not decision.allowed:
            with tracer.span("guardrail_refuse"):
                self.working.append_turn(
                    session_id, ChatTurn(role="assistant", text=decision.message, timestamp=now + 0.001)
                )
            return TurnResult(answer=decision.message, mode="refused", refused=True,
                              trace=tracer.as_dicts())

        # --- intent: operational template vs hybrid KB ---
        with tracer.span("retrieve") as sp:
            intent = match_intent(user_text, now=now, patient_ref=state.patient_ref)
            if intent:
                name, params = intent
                rows = run_template(self.driver, name, **params)
                kb_context, citations = _render_operational(name, rows), [f"graph:{name}"]
                mode, declined = "operational", False
            else:
                mode = "hybrid"
                result = self.hybrid.retrieve(user_text, mode="hybrid")
                kb_context, citations = _render_blocks(result)
                relevant = bool(result.relationships) or (
                    bool(result.passages) and _best_overlap(user_text, result.passages) >= _MIN_OVERLAP
                )
                declined = not relevant
            sp.attributes.update(mode=mode, declined=declined)

        with tracer.span("build_context"):
            history = "\n".join(f"{t.role}: {t.text}" for t in state.turns[-_MAX_HISTORY:])
            episodes = self.episodic.recall(user_text, k=3, patient_ref=state.patient_ref)
            episodic_ctx = "\n".join(f"- {e.text}" for e in episodes) or "(none)"
            prompt = (
                f"## Conversation so far\n{history}\n\n"
                f"## Relevant past interactions\n{episodic_ctx}\n\n"
                f"{kb_context}\n\nQuestion: {user_text}\nAnswer:"
            )

        # --- output policy: answer / decline / escalate ---
        action = self.guardrails.decide_output(user_text, declined=declined)
        escalated = action == "escalate"
        if action == "escalate":
            answer_text = self.guardrails.refusal_for_emergency
        elif action == "decline":
            answer_text = _DECLINE
        else:
            with tracer.span("generate") as sp:
                answer_text, failed = self._generate(prompt)  # de-id'd + retried inside
                sp.attributes["llm_failed"] = failed

        with tracer.span("persist"):
            self.working.append_turn(
                session_id, ChatTurn(role="assistant", text=answer_text, timestamp=now + 0.001)
            )
            self.episodic.add(
                f"Q: {user_text}\nA: {answer_text}", session_id=session_id,
                patient_ref=state.patient_ref, timestamp=now,
            )

        # what the model actually received, after de-identification (for audit/verification)
        model_input = self.llm.deidentifier.deidentify(prompt)
        return TurnResult(
            answer=answer_text, citations=list(dict.fromkeys(citations)),
            mode=mode, declined=(action == "decline"), escalated=escalated,
            model_input=model_input, trace=tracer.as_dicts(),
        )
