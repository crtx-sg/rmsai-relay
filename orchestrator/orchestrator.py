"""Text orchestrator — the stateful chat state machine (Phase 4).

Per turn: load state (working memory) -> classify intent -> retrieve (operational template OR
hybrid KB) + recall episodic -> build a labelled, de-identified context -> LLM -> respond ->
persist (working + episodic). The LLM is always a `DeidentifyingLLM`, so PHI is scrubbed before
any model call by construction.

This is a hand-rolled node pipeline; LangGraph can wrap these same nodes later (the working-memory
checkpointer and node boundaries are already in place).
"""

from __future__ import annotations

import re
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
# "say that again" — re-voice the previous answer from working memory (basic conversational memory).
_REPEAT = re.compile(
    r"\b(repeat( that| it| again| the (last|previous|prior)[\w ]*)?|say (that|it) again|"
    r"come again|what did you (just )?say|(last|previous|prior) (response|answer|reply))\b",
    re.IGNORECASE,
)
_NOTHING_TO_REPEAT = "There is nothing to repeat yet."
_LLM_FALLBACK = "I'm having trouble generating a response right now; please try again."
_MAX_HISTORY = 6

# Keep answers crisp: this is a clinical relay heard over the phone / read in a chat box, so the
# response text IS the payload. Instruct the model to lead with the answer and drop the filler small
# models love to add (preambles, restating the question, disclaimers, "let me know if…").
_ANSWER_INSTRUCTIONS = (
    "You are a clinical relay assistant answering a clinician over voice or chat. Using ONLY the "
    "facts in the context below, answer the question directly in one or two short sentences. State "
    "the specific values, patient IDs (e.g. PT4543), and bed labels exactly as given in the context "
    "— do not say information is unavailable when it is listed. Do NOT greet, restate the question, "
    "explain your reasoning, add disclaimers, mention internal field/table/report names, or offer "
    "further help. Only if the context contains no matching records, say so in one short sentence."
)


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
    """Human-readable scalar: trim float noise, render epoch timestamps, drop event-name underscores."""
    if key in _TS_KEYS and isinstance(value, (int, float)):
        from datetime import datetime, timezone  # noqa: PLC0415

        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, float):
        return f"{value:g}"  # 171.0 -> "171", 97.9 -> "97.9"
    # Event types / conditions are SNAKE_CASE ("AV_BLOCK_2_TYPE2"); TTS would read the underscores
    # aloud, so render them as spaces ("AV BLOCK 2 TYPE2"). Only these fields carry underscores.
    return str(value).replace("_", " ")


def _render_operational(name: str, rows: list[dict]) -> str:
    """Render template rows as readable text (not raw dicts), so the answer reads as prose.

    Each row becomes `- key: value, key: value` with nulls dropped and floats/timestamps tidied.
    Feeding the LLM clean text (instead of `{'hr': 171.0, ...}`) keeps the grounded answer from
    echoing a Python object back to the clinician.
    """
    if not rows:
        return "## Matching records (patient knowledge graph)\n(no matching records)"
    lines = []
    for row in rows:
        pairs = [f"{k}: {_fmt_value(k, v)}" for k, v in row.items() if v is not None]
        lines.append(f"- {', '.join(pairs)}")
    # No internal template name in the header — it leaks into the model's answer as snake_case noise.
    return "## Matching records (patient knowledge graph)\n" + "\n".join(lines)


# Vitals snapshot fields, in spoken order. `spo2` is spelled "S P O 2" so TTS says the letters
# instead of "spo-two"; the rest read fine as-is.
_VITAL_KEYS = ("hr", "sbp", "dbp", "spo2", "rr", "temp")
_SPOKEN_LABEL = {
    "spo2": "S P O 2", "mews_risk": "MEWS risk", "mews": "MEWS", "event_type": "event type",
    "reported_event": "reported event", "false_positive": "false positive",
    "actual_condition": "actual condition",
}


def _label(key: str) -> str:
    return _SPOKEN_LABEL.get(key, key.replace("_", " "))


def _row_to_sentence(row: dict) -> str:
    """Render one graph row as a spoken-friendly sentence: 'At <ts>, patient <id> on bed … had …'."""
    r = {k: v for k, v in row.items() if v is not None}
    lead = []
    ts = r.pop("ts", None) or r.pop("timestamp", None)
    if ts is not None:
        lead.append(f"At {_fmt_value('ts', ts)},")
    if (patient := r.pop("patient", None)):
        lead.append(f"patient {patient}")
    if (bed := r.pop("bed", None)):
        lead.append(f"on bed {bed}")
    if (unit := r.pop("unit", None)):
        lead.append(f"in unit {unit}")
    event = r.pop("event_type", None) or r.pop("event", None) or r.pop("reported_event", None)
    if event:
        lead.append(f"had an event type {_fmt_value('event', event)}")  # underscores -> spaces
    text = " ".join(lead)

    # Artifact refs (ECG strip / raw signal) and HR history get spoken-friendly phrasing, not a raw
    # path/URI or a bare number list read aloud — pull them out before the generic clause loop.
    ecg_plot = r.pop("ecg_plot", None) or r.pop("ecg_plot_ref", None)
    has_signal = r.pop("signal_ref", None) is not None
    hr_history = r.pop("hr_history", None)
    r.pop("hr_history_ts", None)

    clauses = []
    if (crit := r.pop("criticality", None)):
        clauses.append(f"criticality {crit}")
    mews = r.pop("mews_risk", None)
    if mews is None:
        mews = r.pop("mews", None)
    if mews:
        clauses.append(f"MEWS risk {mews}")
    for k in [k for k in r if k not in _VITAL_KEYS]:  # status / actual_condition / action / …
        clauses.append(f"{_label(k)} {_fmt_value(k, r.pop(k))}")
    if clauses:
        text += ("; " if text else "") + "; ".join(clauses)

    if hr_history:
        vals = ", ".join(_fmt_value("hr", v) for v in hr_history)
        text += (f". HR trended from {_fmt_value('hr', hr_history[0])} to "
                 f"{_fmt_value('hr', hr_history[-1])} over {len(hr_history)} readings "
                 f"({_trend_word(hr_history)}): {vals}")
    if ecg_plot:
        text += ". An ECG strip image is available"   # path is in the graph; companion app fetches it
    elif has_signal:
        text += ". The raw ECG is archived"

    vitals = [f"{_SPOKEN_LABEL.get(k, k)} {_fmt_value(k, r[k])}" for k in _VITAL_KEYS if k in r]
    if vitals:
        text += ". The vitals at this event were " + "; ".join(vitals)

    text = text.strip().strip(";").strip()
    return (text + ".") if text else "a record with no details."


def _trend_word(values: list) -> str:
    """Coarse trend direction from the first vs last reading."""
    if len(values) < 2:
        return "single reading"
    delta = values[-1] - values[0]
    return "stable" if abs(delta) < 1 else ("rising" if delta > 0 else "falling")


def _answer_operational(rows: list[dict] | None) -> str:
    """Deterministic, spoken-friendly answer for a structured graph result (no LLM).

    Operational queries return exact rows, so we voice/print them directly — accurate every time, no
    model latency or hallucination. Each row becomes a natural sentence; multiple rows are one
    sentence per line so TTS pauses between them (a hard pause would need backend-specific SSML).
    """
    rows = rows or []
    if not rows:
        return "No matching records."
    sentences = [_row_to_sentence(r) for r in rows]
    if len(sentences) == 1:
        return sentences[0]
    return f"{len(sentences)} matching records.\n" + "\n".join(sentences)


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
        episodic_recall: bool = False,
    ) -> None:
        self.working = working
        self.hybrid = hybrid
        self.episodic = episodic
        self.llm = llm
        self.driver = driver
        self.guardrails = guardrails or Guardrails()
        self.llm_retries = llm_retries
        self.episodic_recall = episodic_recall  # gate recalled "past interactions" in free-text answers

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

        # --- "repeat that" -> re-voice the last answer from working memory (no KB/LLM) ---
        if _REPEAT.search(user_text):
            with tracer.span("repeat"):
                last = next((t.text for t in reversed(state.turns) if t.role == "assistant"), None)
                answer_text = last or _NOTHING_TO_REPEAT
                self.working.append_turn(
                    session_id, ChatTurn(role="assistant", text=answer_text, timestamp=now + 0.001)
                )
            return TurnResult(answer=answer_text, mode="repeat", trace=tracer.as_dicts())

        # --- intent: operational template vs hybrid KB ---
        operational_rows: list[dict] | None = None
        with tracer.span("retrieve") as sp:
            intent = match_intent(user_text, now=now, patient_ref=state.patient_ref)
            if intent:
                name, params = intent
                operational_rows = run_template(self.driver, name, **params)
                kb_context = _render_operational(name, operational_rows)
                citations = [f"graph:{name}"]
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

        # --- output policy: answer / decline / escalate ---
        action = self.guardrails.decide_output(user_text, declined=declined)
        escalated = action == "escalate"
        model_input = ""  # what the model received, de-identified (audit); "" when no model is called
        if action == "escalate":
            answer_text = self.guardrails.refusal_for_emergency
        elif action == "decline":
            answer_text = _DECLINE
        elif mode == "operational":
            # Structured graph result -> deterministic, crisp answer. No LLM is called, so NO prompt
            # is built: conversation history and recalled past interactions never touch an operational
            # answer. (They are only useful for free-text follow-ups, below.)
            answer_text = _answer_operational(operational_rows)
            model_input = self.llm.deidentifier.deidentify(kb_context)  # the rows that informed it
        else:
            # Free-text/hybrid: the LLM needs conversation history (multi-turn follow-ups). Built ONLY
            # here — only when the model is actually invoked. Recalled "past interactions" are added
            # only when `episodic_recall` is on (off by default: grounded in live KB + this chat only).
            with tracer.span("build_context"):
                history = "\n".join(f"{t.role}: {t.text}" for t in state.turns[-_MAX_HISTORY:])
                blocks = [_ANSWER_INSTRUCTIONS, f"## Conversation so far\n{history}"]
                if self.episodic_recall:
                    episodes = self.episodic.recall(user_text, k=3, patient_ref=state.patient_ref)
                    episodic_ctx = "\n".join(f"- {e.text}" for e in episodes) or "(none)"
                    blocks.append(f"## Relevant past interactions\n{episodic_ctx}")
                blocks.append(f"{kb_context}\n\nQuestion: {user_text}\nAnswer:")
                prompt = "\n\n".join(blocks)
            with tracer.span("generate") as sp:
                answer_text, failed = self._generate(prompt)  # de-id'd + retried inside
                sp.attributes["llm_failed"] = failed
            model_input = self.llm.deidentifier.deidentify(prompt)

        with tracer.span("persist"):
            self.working.append_turn(
                session_id, ChatTurn(role="assistant", text=answer_text, timestamp=now + 0.001)
            )
            if self.episodic_recall:  # only archive Q&A when it will actually be recalled
                self.episodic.add(
                    f"Q: {user_text}\nA: {answer_text}", session_id=session_id,
                    patient_ref=state.patient_ref, timestamp=now,
                )

        return TurnResult(
            answer=answer_text, citations=list(dict.fromkeys(citations)),
            mode=mode, declined=(action == "decline"), escalated=escalated,
            model_input=model_input, trace=tracer.as_dicts(),
        )
