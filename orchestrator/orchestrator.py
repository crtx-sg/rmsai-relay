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
from kb.graph.driver import GraphDriver
from kb.graph.lookup import match_intent
from kb.graph.templates import run_template
from kb.hybrid.retriever import HybridRetriever
from kb.vector.answer import _best_overlap
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory

_MIN_OVERLAP = 0.18
_DECLINE = "I don't have information on that in the knowledge base."
_MAX_HISTORY = 6


@dataclass
class TurnResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    mode: str = "hybrid"
    declined: bool = False
    model_input: str = ""  # the (de-identified) text actually sent to the model — for audit/tests


def _render_blocks(result) -> tuple[str, list[str]]:
    passages = "\n\n".join(f"[P{i + 1}] ({p.source}) {p.text}" for i, p in enumerate(result.passages))
    rels = "\n".join(f"[R{i + 1}] ({r.source}) {r.fact}" for i, r in enumerate(result.relationships))
    text = f"## Retrieved passages\n{passages or '(none)'}\n\n## Known relationships\n{rels or '(none)'}"
    cites = [p.source for p in result.passages] + [r.source for r in result.relationships]
    return text, cites


def _render_operational(name: str, rows: list[dict]) -> str:
    body = "\n".join(f"- {r}" for r in rows) or "(no matching records)"
    return f"## Operational result: {name}\n{body}"


class Orchestrator:
    def __init__(
        self,
        working: WorkingMemory,
        hybrid: HybridRetriever,
        episodic: EpisodicMemory,
        llm: DeidentifyingLLM,
        driver: GraphDriver,
    ) -> None:
        self.working = working
        self.hybrid = hybrid
        self.episodic = episodic
        self.llm = llm
        self.driver = driver

    def handle_turn(
        self, session_id: str, user_text: str, *, patient_ref: str | None = None, now: float | None = None
    ) -> TurnResult:
        now = now if now is not None else time.time()
        state = self.working.get_or_create(session_id)
        if patient_ref:
            state.patient_ref = patient_ref
            self.working.save(state)
        self.working.append_turn(session_id, ChatTurn(role="clinician", text=user_text, timestamp=now))

        # --- intent: operational template vs hybrid KB ---
        intent = match_intent(user_text, now=now)
        declined = False
        if intent:
            name, params = intent
            rows = run_template(self.driver, name, **params)
            kb_context, citations = _render_operational(name, rows), [f"graph:{name}"]
            mode = "operational"
        else:
            mode = "hybrid"
            result = self.hybrid.retrieve(user_text, mode="hybrid")
            kb_context, citations = _render_blocks(result)
            relevant = bool(result.relationships) or (
                bool(result.passages) and _best_overlap(user_text, result.passages) >= _MIN_OVERLAP
            )
            declined = not relevant

        # --- memory context ---
        history = "\n".join(f"{t.role}: {t.text}" for t in state.turns[-_MAX_HISTORY:])
        episodes = self.episodic.recall(user_text, k=3, patient_ref=state.patient_ref)
        episodic_ctx = "\n".join(f"- {e.text}" for e in episodes) or "(none)"

        prompt = (
            f"## Conversation so far\n{history}\n\n"
            f"## Relevant past interactions\n{episodic_ctx}\n\n"
            f"{kb_context}\n\nQuestion: {user_text}\nAnswer:"
        )

        if declined:
            answer_text = _DECLINE
        else:
            # DeidentifyingLLM scrubs the prompt before the model sees it (fail closed).
            answer_text = self.llm.generate(prompt)

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
            mode=mode, declined=declined, model_input=model_input,
        )
