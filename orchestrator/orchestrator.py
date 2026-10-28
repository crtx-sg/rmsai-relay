"""Text orchestrator — the stateful chat state machine (Phase 4).

Per turn: load state (working memory) -> classify intent -> retrieve (operational template OR
hybrid KB) + recall episodic -> build a labelled, de-identified context -> LLM -> respond ->
persist (working + episodic). The LLM is always a `DeidentifyingLLM`, so PHI is scrubbed before
any model call by construction.

This is a hand-rolled node pipeline; LangGraph can wrap these same nodes later (the working-memory
checkpointer and node boundaries are already in place).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from functools import partial

from common.config import DEFAULT
from common.event_types import NORMAL_SINUS, event_type_from_text
from common.providers import DeidentifyingLLM
from common.redacting_logger import get_redacting_logger
from common.schemas import ChatTurn
from common.tracing import Tracer
from common.vitals_format import fmt_vital
from kb.graph.driver import GraphDriver
from kb.graph.events import is_pending
from kb.graph.llm_router import route as llm_route
from kb.graph.spoken import normalize_spoken_query
from kb.graph.lookup import (
    asks_event_protocol, is_patient_specific, looks_like_event_listing, match_intent,
    names_protocol,
    names_bed_label, names_unresolved_bed,
)
from kb.graph.templates import EVENT_HISTORY, describe_event_history, run_template
from kb.hybrid.answer import _MIN_OVERLAP, is_relevant
from kb.hybrid.retriever import HybridRetriever
from kb.vector.answer import _best_overlap
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory

from . import protocol_answer
from .report import spoken_bed, spoken_event
from .guardrails import UNGROUNDED_ANSWER, Guardrails, is_emergency, ungrounded_identifiers

def explain_decline(result, overlap: float, top_score: float = 0.0, *,
                    min_relevance: float = 0.60, min_overlap: float = _MIN_OVERLAP) -> str:
    """Why this turn declined, and the lever that would fix it. Empty string if it shouldn't have.

    A decline has exactly three causes, and they need opposite responses — an empty corpus needs a
    document, a rejected passage needs a threshold or a better retriever, and a wrong-topic passage
    needs nothing at all (the decline was correct). Without this the clinician and the log both just
    see "I don't have information on that", which reads as a broken knowledge base every time.
    """
    if result.relationships:
        return ""
    if not result.passages:
        return ("retrieval returned nothing — no vector passages and no graph relationships. The "
                "corpus has no document on this topic: add one under docs/ and re-index "
                "(cli.kb_vector index).")
    top = result.passages[0]
    return (
        f"{len(result.passages)} passage(s) retrieved, but neither relevance signal vouched for "
        f"them: semantic similarity {top_score:.2f} < {min_relevance:.2f} AND word overlap "
        f"{overlap:.2f} < {min_overlap:.2f} (top: {top.source}). If that passage is ON-topic, the "
        f"gate is too strict — lower KB_MIN_RELEVANCE; note the similarity arm only discriminates "
        f"under EMBEDDER=bge (hashing scores on- and off-topic queries alike). If it is off-topic, "
        f"the decline was correct and the corpus is missing this subject."
    )


_DECLINE = "I don't have information on that in the knowledge base."
# An event listing no rule could resolve. Declined rather than guessed, so every event list the
# agent does give comes from the same query the Events tab and `cli.events` run.
_EVENT_LISTING_HINT = (
    "I can only list events I can show you on the dashboard: by patient (a PT number, or this "
    "patient), by bed number, by event type, or by status (pending, acknowledged, withheld). For "
    "example: pending events on bed 2.")
# A bed the clinician named that did not resolve to a number. Never handed to the LLM router: live,
# "last reported event on vid nine" was routed to the call's own event and read out bed 2's vitals.
_BED_HINT = ("I couldn't tell which bed you meant. Please say the bed number again, for example: "
             "bed seven.")
# A question about this patient/event that no graph query covers. Never answered by free generation
# over the documents: live, the model invented "P3", ages, beds and event IDs from archived reports.
_PATIENT_HINT = ("I can tell you this patient's rhythm, vitals, MEWS, heart-rate trend, history, "
                 "medications, events and bed, and why the alert was raised. Which would you like?")
# A router pick about "the patient's latest event", when the session is about a specific (called or
# selected) event: answer about that event. The patient can have a newer one (VT after the VF call).
_CALLED_EVENT_TWIN = {
    "vitals_at_patient_last_event": "vitals_at_selected_event",
    "mews_at_patient_last_event": "mews_at_selected_event",
    "hr_trend_for_patient_last_event": "hr_trend_for_event",
    "ecg_strips_for_patient_last_event": "ecg_strip_for_event",
}
# Words that say what KIND of question it is, not what it is about: the rest must appear in a
# retrieved passage (unless similarity alone is high). "How do you treat sepsis?" matched a VT
# passage on "treat" and was answered from it, with no sepsis document in the corpus.
_GENERIC_WORDS = frozenset(
    "about after also anyone before does done during from have here into like long more most much "
    "only other over same some such than that their them then there these they this those very what "
    "when where which while will with would your should could tell explain mean means meaning "
    "patient patients treat treated treating treatment treatments manage managed managing "
    "management handle handling protocol protocols guideline guidelines dose doses dosing give "
    "given says said steps step procedure procedures recommended recommendation".split())
_TOPIC_SIMILARITY = 0.70
# "What does a MEWS of 5 mean?": answered from the scorer's band + the document's escalation rule.
# The 3B model gave "a high level of distress".
# The number must be the score itself ("MEWS of 5", "a MEWS score is 7"): "MEWS escalation in 30
# minutes means?" read 30 as a MEWS. A named bed ("is bed 4's MEWS significant?") is a lookup.
_MEWS_SCORE = re.compile(r"\b(?:mews|news)\s*(?:score\s*)?(?:of|is|=|at)?\s*(\d{1,2})\b",
                         re.IGNORECASE)
_MEANING = re.compile(r"\b(?:mean\w*|indicat\w*|signif\w*|interpret\w*)\b", re.IGNORECASE)
_MEWS_DOC = "mews_escalation.md"


def _missing_topic(question: str, passages) -> list[str]:
    """The question's topic words, when none of them appears in any retrieved passage; else []."""
    terms = [w for w in re.findall(r"[a-z][a-z0-9-]{3,}", question.lower())
             if w not in _GENERIC_WORDS]
    if not terms:
        return []
    text = " ".join(f"{p.text} {p.source}".lower() for p in passages)
    return [] if any(t[:5] in text for t in terms) else terms


# "say that again" — re-voice the previous answer from working memory (basic conversational memory).
_REPEAT = re.compile(
    r"\b(repeat( that| it| again| the (last|previous|prior)[\w ]*)?|say (that|it) again|"
    r"come again|what did you (just )?say|(last|previous|prior) (response|answer|reply))\b",
    re.IGNORECASE,
)
_NOTHING_TO_REPEAT = "There is nothing to repeat yet."
# Closing pleasantries: answered directly, never searched in the KB (which declined "thank you").
# A whole utterance made only of courtesy phrases ("okay, thank you", "that's all, bye").
_COURTESY = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|great|cheers|thanks?(?: you)?(?: (?:very|so) much)?|thank you|"
    r"(?:good)?bye(?: bye)?|that'?s (?:all|it)|that is all|nothing else|i'?m done)\W*)+$",
    re.IGNORECASE)
_FAREWELL = re.compile(r"\b(?:(?:good)?bye|that'?s (?:all|it)|that is all|nothing else|i'?m done)\b",
                       re.IGNORECASE)
_THANKS = re.compile(r"\b(?:thanks?|thank you|cheers)\b", re.IGNORECASE)
_YOU_ARE_WELCOME = "You're welcome."
_GOODBYE_REPLY = "Goodbye."
_LLM_FALLBACK = "I'm having trouble generating a response right now; please try again."
# De-id placeholders the scrubbers emit (<PERSON>, <NAME>, <PHONE>, <US_DRIVER_LICENSE>, ...).
_PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]{2,}>")
_PLACEHOLDER_ANSWER = ("I can't repeat identifying details from that question. "
                       "Please refer to the patient by ID or bed.")
_log = get_redacting_logger("rmsai.orchestrator")
_MAX_HISTORY = 6
# A question that leans on the conversation ("what about its dose?", "and for that one?").
_REFERS_BACK = re.compile(r"\b(?:it|its|that|those|them|they|their|same|what about|how about|"
                          r"instead|also|else|above|previous|earlier)\b", re.IGNORECASE)

# Keep answers crisp: this is a clinical relay heard over the phone / read in a chat box, so the
# response text IS the payload. Instruct the model to lead with the answer and drop the filler small
# models love to add (preambles, restating the question, disclaimers, "let me know if…").
_ANSWER_INSTRUCTIONS = (
    "You are a clinical relay assistant answering a clinician over voice or chat. Using ONLY the "
    "facts in the context below, answer the question directly in one or two short sentences. State "
    "the specific values, patient IDs, and bed labels exactly as given in the context "
    "— do not say information is unavailable when it is listed. Do NOT greet, restate the question, "
    "explain your reasoning, add disclaimers, mention internal field/table/report names, or offer "
    "further help. Only if the context contains no matching records, say so in one short sentence. "
    "[P1], [P2], [R1]… label the passages and relationships for citation: they are never patient "
    "IDs, so never call anything 'patient P1'."
)
# Read aloud: "[P2] and [P3] indicate…" became "and indicate…" once the labels were stripped.
_SPOKEN_INSTRUCTIONS = (
    " This answer is read aloud: write plain sentences with no passage labels, document names, "
    "markdown or lists."
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
    # Set when the answer is an event listing: the Events-tab filter that shows the same rows
    # (patient/bed/event_type/status/min_criticality/since), so the app can link to it.
    history_filter: dict | None = None


def history_filter_for(params: dict) -> dict:
    """Template params -> the `/events` request fields that reproduce the same listing."""
    out = {"status": params.get("status") or "any"}
    for src, dst in (("patient_id", "patient"), ("bed", "bed"), ("event_type", "event_type"),
                     ("min_criticality", "min_criticality")):
        if params.get(src):
            out[dst] = params[src]
    if params.get("since"):
        out["since"] = str(params["since"])  # epoch seconds: /events accepts it as-is
    return out


def _is_patient_passage(source: str) -> bool:
    """Vector passages that come from a patient's event report (`report:<uuid>#…`), not a document.

    `cli.consume` archives event-report narratives into the same collection as the SOPs, so a
    retrieved passage may carry patient data: those stay de-identified.
    """
    return source.startswith("report:")


def _render_block_parts(result) -> tuple[list[tuple[str, bool]], list[str]]:
    """The retrieved blocks as `(text, sensitive)` prompt parts, plus citations.

    Same text as before, split so only PHI-capable content is de-identified: patient-report
    passages and graph facts are sensitive; SOP/policy passages and the markers are not.
    """
    parts: list[tuple[str, bool]] = [("## Retrieved passages\n", False)]
    if not result.passages:
        parts.append(("(none)", False))
    for i, p in enumerate(result.passages):
        if i:
            parts.append(("\n\n", False))
        parts.append((f"[P{i + 1}] ({p.source}) ", False))
        parts.append((p.text, _is_patient_passage(p.source)))
    parts.append(("\n\n## Known relationships\n", False))
    if not result.relationships:
        parts.append(("(none)", False))
    for i, r in enumerate(result.relationships):
        if i:
            parts.append(("\n", False))
        parts.append((f"[R{i + 1}] ({r.source}) ", False))
        parts.append((r.fact, True))
    cites = [p.source for p in result.passages] + [r.source for r in result.relationships]
    return parts, cites


def _render_blocks(result) -> tuple[str, list[str]]:
    parts, cites = _render_block_parts(result)
    return "".join(t for t, _ in parts), cites


_TS_KEYS = {"ts", "timestamp", "due", "due_at", "generated_at"}


def _fmt_value(key: str, value) -> str:
    """Human-readable scalar: trim float noise, render epoch timestamps, drop event-name underscores."""
    if key in _TS_KEYS and isinstance(value, (int, float)):
        from datetime import datetime, timezone  # noqa: PLC0415

        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if key in _VITAL_KEYS and isinstance(value, (int, float)):
        return fmt_vital(key, value)  # integers; temperature to one decimal
    if isinstance(value, float):
        return f"{value:g}"  # 171.0 -> "171", 97.9 -> "97.9"
    # Event types / conditions are SNAKE_CASE ("AV_BLOCK_2_TYPE2"); TTS would read the underscores
    # aloud, so render them as spaces ("AV BLOCK 2 TYPE2"). Only these fields carry underscores.
    return str(value).replace("_", " ")


# shortcut: the audit copy of graph rows is cut here, since de-id (~0.5 ms/char) runs before the
# spoken reply; move the audit write off the reply path if full rows must be kept.
_AUDIT_MAX_CHARS = 8000


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
        # why_json is a multi-KB explanation blob (read only for MEWS): rendering it made de-id of
        # a 12-row ranking take ~36 s and the voice reply never arrived.
        pairs = [f"{k}: {_fmt_value(k, v)}" for k, v in row.items()
                 if v is not None and k != "why_json"]
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
    r.pop("alert_gate", None)  # bookkeeping for the pending count, not something to read out
    r.pop("report_id", None)  # internal id / file path: the app opens the report, nobody reads it
    r.pop("report_uri", None)

    clauses = []
    if (crit := r.pop("criticality", None)):
        clauses.append(f"criticality {crit}")
    mews = r.pop("mews_risk", None)
    if mews is None:
        mews = r.pop("mews", None)
    mews_detail = _mews_from_explanation(r.pop("why_json", None))  # never voiced raw
    if mews_detail:
        clauses.append(mews_detail)
    elif mews:
        clauses.append(f"MEWS risk {mews}")
    # `false_positive` is a bool: voice "flagged as a false positive" only when True; a bare
    # "false positive False" reads as noise, so drop it otherwise.
    if r.pop("false_positive", None) is True:
        clauses.append("flagged as a false positive")
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


def _mews_from_explanation(why_json) -> str | None:
    """`MEWS 6 (High); contributing: Respiratory Rate 25 scores 2, …` from an event's stored
    explanation (`why_json`), or None when it is missing or unreadable (the caller then falls back to
    the MEWS risk alone)."""
    if not why_json:
        return None
    try:
        m = json.loads(why_json)["vitals"]["mews"]
        text = f"MEWS {m['score']} ({m['risk']})" if m["score"] is not None else "MEWS unavailable"
    except (ValueError, KeyError, TypeError):
        return None
    comps = ", ".join(f"{c['name']} {c['value']} scores {c['score']}"
                      for c in m.get("components") or [] if isinstance(c, dict))
    return text + (f"; contributing: {comps}" if comps else "")


# --- spoken (phone) answers for one event record --------------------------------------------------

# Templates whose single row is one event record (rhythm, time, MEWS, vitals): on a call these are
# said as a sentence a listener can follow, not the text form's "hr 195; sbp 43; S P O 2 84".
_RECORD_TEMPLATES = {
    "vitals_at_event", "vitals_at_patient_last_event", "vitals_at_patient_event_of_type",
    "vitals_at_selected_event", "vitals_for_bed_last_event", "mews_at_selected_event",
    "mews_at_patient_last_event", "bed_last_event", "hr_trend_for_event",
    "hr_trend_for_patient_last_event", "ecg_strip_for_event", "ecg_strips_last_event",
    "ecg_strips_for_patient_last_event", "ecg_strips_last_event_of_type", "patient_bed",
}
_SPOKEN_VITAL = {"hr": "heart rate {}", "spo2": "oxygen saturation {} percent",
                 "rr": "respiratory rate {}", "temp": "temperature {} degrees"}
# MEWS component names as the vendored scorer writes them -> words for the ear.
_SPOKEN_COMPONENT = {"systolic bp": "systolic blood pressure", "spo2": "oxygen saturation",
                     "temp": "temperature", "resp rate": "respiratory rate"}
_FOCUS_NAME = {"hr": "Heart rate", "bp": "Blood pressure", "spo2": "Oxygen saturation",
               "rr": "Respiratory rate", "temp": "Temperature"}


def _spoken_vitals(r: dict, focus: str | None = None) -> list[str]:
    out = []
    for key in ("hr", "bp", "spo2", "rr", "temp"):
        if focus and key != focus:
            continue
        if key == "bp":
            if r.get("sbp") is not None and r.get("dbp") is not None:
                out.append(f"blood pressure {_fmt_value('sbp', r['sbp'])} over "
                           f"{_fmt_value('dbp', r['dbp'])}")
            elif r.get("sbp") is not None:
                out.append(f"systolic blood pressure {_fmt_value('sbp', r['sbp'])}")
        elif r.get(key) is not None:
            out.append(_SPOKEN_VITAL[key].format(_fmt_value(key, r[key])))
    return out


_READING_LABEL = {"hr": "heart rate", "sbp": "systolic blood pressure",
                  "dbp": "diastolic blood pressure", "spo2": "oxygen saturation",
                  "rr": "respiratory rate", "temp": "temperature"}


def _answer_vital_readings(row: dict, focus: str | None, n: int) -> str:
    """'Patient PT942338 on bed 1, last 3 readings before the event, oldest first: respiratory rate
    15, 15, 16.' Systolic and diastolic are listed apart: they are sampled at different times."""
    head = f"Patient {row['patient']}" if row.get("patient") else "This patient"
    if row.get("bed"):
        head += f" on {spoken_bed(row['bed'])}"
    keys = {"bp": ("sbp", "dbp")}.get(focus, (focus,)) if focus else _VITAL_KEYS
    parts = []
    for key in keys:
        label = _READING_LABEL[key]
        if hist := row.get(key):
            parts.append(f"{label} {', '.join(_fmt_value(key, v) for v in hist[-n:])}")
        else:
            parts.append(f"{label} not recorded")
    return f"{head}, last {n} readings before the event, oldest first: {'; '.join(parts)}."


def _spoken_time(ts) -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    if t.date() == datetime.now(timezone.utc).date():
        return f"at {t:%H:%M} UTC"
    return f"on {t.day} {t:%B} at {t:%H:%M} UTC"


def _spoken_record(row: dict, focus: str | None = None) -> str:
    """One event record said aloud: 'Patient PT998224 on bed 2: ventricular fibrillation at 07:16
    UTC, Critical criticality, MEWS 9, Critical risk. Vitals: heart rate 195, blood pressure 43 over
    30, …'. `focus` ('mews' or one vital) answers just that part (live 2026-10-14: the full dump was
    read for "what was the MEWS score", and MEWS was given as a band without the score)."""
    r = {k: v for k, v in row.items() if v is not None}
    event = r.get("event_type") or r.get("event") or r.get("reported_event")
    head = f"Patient {r['patient']}" if r.get("patient") else "This patient"
    if r.get("bed"):
        head += f" on {spoken_bed(r['bed'])}"
    head += f": {spoken_event(event)}" if event else ""
    if r.get("ts") is not None:
        head += f" {_spoken_time(r['ts'])}"

    mews = None
    try:
        mews = json.loads(r["why_json"])["vitals"]["mews"] if r.get("why_json") else None
    except (ValueError, KeyError, TypeError):
        mews = None
    if mews and mews.get("score") is not None:
        mews_text = f"MEWS {mews['score']}, {mews.get('risk', '').lower()} risk".rstrip(", ")
    elif r.get("mews_risk") == "Unknown":
        mews_text = "MEWS unavailable"
    elif r.get("mews_risk"):
        mews_text = f"MEWS risk {str(r['mews_risk']).lower()}"
    else:
        mews_text = ""

    if focus == "mews":
        if not mews_text:
            return f"{head}. No MEWS score is recorded for this event."
        comps = [f"{_SPOKEN_COMPONENT.get(str(c['name']).lower(), str(c['name']).lower())} "
                 f"{c['value']} scores {c['score']}"
                 for c in (mews or {}).get("components") or []
                 if isinstance(c, dict) and c.get("score")]
        return f"{head}. {mews_text}." + (f" Contributing: {', '.join(comps)}." if comps else "")
    if r.get("ecg_plot") or r.get("signal_ref"):  # a strip can be shown, not read out
        return (f"{head}. The ECG strip is available in the app." if r.get("ecg_plot")
                else f"{head}. The raw ECG is archived; no strip image was rendered.")
    if r.get("hr_history"):  # a trend answer: the shape, not 25 numbers read aloud
        h = [float(v) for v in r["hr_history"]]
        word = _trend_word(h)
        shape = (f"heart rate was {word} at about {_fmt_value('hr', h[0])}" if word == "stable"
                 else f"heart rate was {word}, from {_fmt_value('hr', h[0])} to "
                      f"{_fmt_value('hr', h[-1])}")
        return (f"{head}. Over the {len(h)} readings before the event, {shape}, peaking at "
                f"{_fmt_value('hr', max(h))}.")
    if focus in _FOCUS_NAME:
        vitals = _spoken_vitals(r, focus)
        if not vitals:
            return f"{head}. {_FOCUS_NAME[focus]} was not recorded for this event."
        return f"{head}. {vitals[0][0].upper()}{vitals[0][1:]}."

    parts = [head]
    if r.get("criticality"):
        parts.append(f"rated {str(r['criticality']).lower()}")
    if mews_text:
        parts.append(mews_text)
    text = ", ".join(parts) + "."
    if (vitals := _spoken_vitals(r)):
        text += " Vitals: " + ", ".join(vitals) + "."
    status = r.get("status")
    if status in ("acknowledged", "resolved"):
        text += f" It has been {status}."
    elif status and r.get("alert_gate"):
        text += " It is awaiting acknowledgment."
    return text


def _protocol_question(event_type) -> str | None:
    """The document question for an event's rhythm: 'What is the protocol for ventricular
    fibrillation?' (None without a rhythm)."""
    return f"What is the protocol for {spoken_event(event_type)}?" if event_type else None


def _speakable(text: str) -> str:
    """Stored explanation text, said aloud: symbols and abbreviations as words."""
    for a, b in (("≥", "at or above"), ("≤", "at or below"), ("SpO₂", "oxygen saturation"),
                 ("SpO2", "oxygen saturation"), (" %", " percent"), ("%", " percent"),
                 ("bpm", "beats per minute"), ("HR ", "heart rate "),
                 ("Systolic BP", "systolic blood pressure"), ("Heart Rate", "heart rate")):
        text = text.replace(a, b)
    return re.sub(r"\bmin\b", "minutes", text)


def _answer_event_summary(row: dict, focus: str | None) -> str:
    """The called event's rhythm / confidence / time / false-positive flag / criticality / reason,
    from that event (never the patient's newest one)."""
    r = {k: v for k, v in row.items() if v is not None}
    who = f"Patient {r['patient']}" if r.get("patient") else "This patient"
    if r.get("bed"):
        who += f" on {spoken_bed(r['bed'])}"
    rhythm = spoken_event(r.get("event_type"))
    conf = f"{float(r['confidence']):.0%}".replace("%", " percent") if r.get("confidence") is not None \
        else None
    when = _spoken_time(r["ts"]) if r.get("ts") is not None else ""
    if focus == "confidence":
        return (f"{who}: the model is {conf} confident the rhythm is {rhythm}." if conf
                else f"{who}: no model confidence is recorded for this event.")
    if focus == "time":
        if r.get("ts") is None:
            return f"{who}: the event time is not recorded."
        mins = int((time.time() - float(r["ts"])) // 60)
        ago = f", about {mins} minutes ago" if 0 <= mins < 180 else ""
        return f"{who}: {rhythm} was detected {when}{ago}."
    if focus == "false_positive":
        if r.get("false_positive"):
            return (f"{who}: yes. The model read the rhythm as {rhythm}"
                    f"{f' with {conf} confidence' if conf else ''}, so the rhythm alarm is treated as "
                    f"a false positive. Any alert was raised on the vitals.")
        return (f"{who}: no. The model detected {rhythm}{f' with {conf} confidence' if conf else ''};"
                f" it is not flagged as a false positive.")
    if focus == "why":
        if r.get("why"):
            return f"{who}: " + _speakable(str(r["why"])).replace("Shown because:", "you were "
                                                                  "alerted because", 1)
        return f"{who}: {rhythm}, rated {str(r.get('criticality') or 'unknown').lower()}."
    if focus == "criticality":
        mews = _mews_from_explanation(r.get("why_json"))
        mews = mews.split(";")[0] if mews else (f"MEWS risk {r['mews_risk']}" if r.get("mews_risk")
                                                else "")
        return (f"{who}: {rhythm} {when}, rated {str(r.get('criticality') or 'unknown').lower()}"
                + (f", {mews}." if mews else "."))
    # rhythm (default)
    return f"{who}: the model detected {rhythm} {when}" + (f", {conf} confident." if conf else ".")


def _answer_action_items(rows: list[dict]) -> str:
    if not rows:
        return "No action items are recorded for this event."
    r0 = rows[0]
    who = f"patient {r0['patient']}" if r0.get("patient") else "this patient"
    if r0.get("bed"):
        who += f" on {spoken_bed(r0['bed'])}"
    items = [str(r["action"]).replace(" — ", ": ") for r in rows if r.get("action")]
    if not items:
        return "No action items are recorded for this event."
    open_n = sum(1 for r in rows if r.get("status") == "outstanding")
    return (f"For {who}, the recommended actions are: " + "; ".join(items) + "."
            + (f" {open_n} still outstanding." if open_n and open_n != len(rows) else ""))


def _answer_event_count(rows: list[dict], window: str | None) -> str:
    r = rows[0] if rows else {}
    n = int(r.get("events") or 0)
    window = window or "in total"
    if not n:
        return f"No events {window}."
    return (f"{n} event{'s' if n != 1 else ''} {window} for {r.get('patients', 0)} "
            f"patient{'s' if r.get('patients') != 1 else ''}: {r.get('alerted', 0)} alerted, "
            f"{r.get('pending', 0)} still awaiting acknowledgment.")


def _answer_mews_meaning(score: int, escalation: str | None) -> str:
    """'What does a MEWS of 5 mean?' — the band from the same scorer that rates the alerts, plus the
    escalation rule quoted from the MEWS document."""
    from ecg_transcovnet.mews import _risk_level  # noqa: PLC0415 - vendored scorer's banding

    text = f"A MEWS of {score} is {_risk_level(score).lower()} risk."
    return f"{text} {escalation.strip()}" if escalation else text


_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
_SPOKEN_PATIENTS_MAX = 10  # a phone list longer than this is cut, with a count of the rest


def _mews_score(why_json) -> int | None:
    try:
        return int(json.loads(why_json)["vitals"]["mews"]["score"])
    except (TypeError, ValueError, KeyError):
        return None


def _answer_patient_ranking(rows: list[dict], *, order: str | None = None,
                            spoken: bool = False) -> str:
    """Every patient, sickest first (or in bed order): open alerts by their most severe criticality,
    then the latest MEWS score, then the newest. One short line each: bed, patient, criticality,
    rhythm, MEWS, open alerts."""
    if not rows:
        return "No patients have recorded events."

    def _key(r):
        if order == "bed":
            m = re.search(r"(\d+)$", str(r.get("bed") or ""))
            return (0, int(m.group(1))) if m else (1, 0)
        return (-_RANK.get(r.get("open_criticality") or "", -1), -(_mews_score(r.get("why_json")) or -1),
                -(r.get("ts") or 0))

    rows = sorted(rows, key=_key)

    def _line(r) -> str:
        bed = spoken_bed(r.get("bed")) if spoken else f"bed {r.get('bed') or 'unassigned'}"
        who = f"{bed[0].upper()}{bed[1:]}, {r.get('patient')}"
        score = _mews_score(r.get("why_json"))
        mews = f"MEWS {score}" if score is not None else (
            f"MEWS risk {str(r['mews_risk']).lower()}" if r.get("mews_risk") else "")
        n = int(r.get("open_alerts") or 0)
        if n:
            parts = [str(r.get("open_criticality") or "").lower(), spoken_event(r.get("open_event")),
                     mews, f"{n} open alert{'s' if n != 1 else ''}"]
        else:
            parts = ["no open alerts", f"latest {spoken_event(r.get('latest_event'))}", mews]
        return f"{who}: " + ", ".join(p for p in parts if p)

    head = f"{len(rows)} patient{'s' if len(rows) != 1 else ''}, " + (
        "in bed order" if order == "bed" else "sickest first")
    shown = rows[:_SPOKEN_PATIENTS_MAX] if spoken else rows
    if spoken:
        text = f"{head}. " + " ".join(f"{i}. {_line(r)}." for i, r in enumerate(shown, 1))
        rest = len(rows) - len(shown)
        return text + (f" And {rest} more; ask for the rest by bed." if rest else "")
    return f"{head}:\n" + "\n".join(f"{i}. {_line(r)}" for i, r in enumerate(shown, 1))


def _spoken_list(items: list[str]) -> str:
    items = [str(i) for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _answer_clinical_history(rows: list[dict], focus: str | None = None) -> str:
    """A patient's clinical background (diagnoses, medications, surgeries, symptoms) from the graph.
    Pseudonym only. Abbreviated condition names are expanded ("MI" -> myocardial infarction)."""
    from kb.graph.entities import canonical_condition  # noqa: PLC0415

    if not rows:
        return "No clinical history is recorded for this patient."
    r = rows[0]
    pt = r.get("patient")
    who = f"patient {pt}" if pt else "this patient"
    conditions = [canonical_condition(c) if canonical_condition(c) != c.strip().lower() else c
                  for c in r.get("conditions") or [] if c]
    sections = {
        "conditions": ("History", conditions),
        "medications": ("Medications", r.get("medications") or []),
        "surgeries": ("Prior surgeries", r.get("surgeries") or []),
        "symptoms": ("Recorded symptoms", r.get("symptoms") or []),
    }
    if focus == "allergies":
        return f"Allergies are not recorded in this system for {who}."
    sex = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(
        str(r.get("gender") or "").lower())
    if focus == "demographics":
        facts = [f"{r['age']} years old" if r.get("age") else None, sex]
        facts = [f for f in facts if f]
        if not facts:
            return f"Age and sex are not recorded for {who}."
        return f"{who[0].upper()}{who[1:]} is {', '.join(facts)}."
    if focus in sections:
        label, items = sections[focus]
        if not items:
            return f"No {label.lower()} are recorded for {who}."
        return f"{who[0].upper()}{who[1:]}. {label}: {_spoken_list(items)}."
    who = f"{who[0].upper()}{who[1:]}"
    if r.get("age") and sex:
        who += f", {r['age']} year old {sex}"
    lines = [f"{label}: {_spoken_list(items)}." for label, items in sections.values() if items]
    return f"{who}. " + (" ".join(lines) if lines else "No clinical history is recorded.")


def _trend_word(values: list) -> str:
    """Coarse trend direction from the first vs last reading."""
    if len(values) < 2:
        return "single reading"
    delta = values[-1] - values[0]
    return "stable" if abs(delta) < 1 else ("rising" if delta > 0 else "falling")


# The "who/where" fields hoisted into the multi-row header when every row shares them. Deliberately
# only these (not arbitrary shared fields) so the header stays a clean context line, never noise like
# "false positive False".
_CONTEXT_KEYS = ("patient", "bed", "unit")


def _describe_context(shared: dict) -> str:
    """Readable noun phrase for fields common to every row: 'patient PT3561, bed Unit1-Bed01, unit …'."""
    return ", ".join(f"{_label(k)} {_fmt_value(k, shared[k])}" for k in _CONTEXT_KEYS if k in shared)


# What an empty result means, per template, instead of a bare "No matching records."
_EMPTY_ANSWERS = {
    "worklist": "Your worklist is clear: no unacknowledged events.",
    "comorbidity_neighborhood": "No co-morbidity pattern is recorded for that condition among "
                                "current patients.",
    "alarm_counts": "There are no alarms on record.",
}


def _times(n: int) -> str:
    return "" if n == 1 else " twice" if n == 2 else f" {n} times"


def _spoken_summary(rows: list[dict], name: str | None) -> str:
    """A short spoken answer for a multi-row graph result (phone/voice turns only).

    Read aloud, the full list (8 records, each with a timestamp) ran for half a minute; the caller
    talked over it and re-asked. Text chat keeps the full `_answer_operational` list.
    """
    rows = [{k: v for k, v in r.items() if v is not None} for r in rows]
    n = len(rows)
    if name == "worklist":
        parts = [f"{n} unacknowledged events."]
        for level in ("Critical", "High", "Medium", "Low"):
            tier = [r for r in rows if r.get("criticality") == level]
            if not tier:
                continue
            counts: dict[tuple, int] = {}
            for r in tier:  # rows arrive most recent first; keep that order
                key = (spoken_bed(r.get("bed")), spoken_event(r.get("event")))
                counts[key] = counts.get(key, 0) + 1
            items = [f"{bed} {event}{_times(c)}" for (bed, event), c in counts.items()]
            parts.append(f"{len(tier)} {level}: {', '.join(items)}.")
        parts.append("Ask about a bed for details.")
        return " ".join(parts)
    if name == "alarm_counts":
        top = rows[0].get("alarms", 0)
        leaders = [r for r in rows if r.get("alarms") == top]
        who = [f"{spoken_bed(r.get('bed'))}, patient {r.get('patient')}" for r in leaders]
        listed = who[0] if len(who) == 1 else ", ".join(who[:-1]) + " and " + who[-1]
        each = " each" if len(leaders) > 1 else ""
        text = f"Most alarms: {top}{each}, on {listed}."
        rest = [r for r in rows if r.get("alarms") != top]
        if rest:
            by_n: dict[int, list[str]] = {}
            for r in rest:
                by_n.setdefault(r.get("alarms", 0), []).append(spoken_bed(r.get("bed")))
            text += " " + " ".join(f"{c} each on {_spoken_list(beds)}." if len(beds) > 1
                                   else f"{c} on {beds[0]}." for c, beds in by_n.items())
        return text
    if name == "patients_with_event_type":
        who = [f"{r.get('patient')} on {spoken_bed(r.get('bed'))}" for r in rows]
        return f"{n} patients: {_spoken_list(who)}."
    if name == "outstanding_action_items":
        high = [r for r in rows if r.get("priority") == "high"]
        beds = {r.get("bed") for r in rows if r.get("bed")}
        top = [f"{spoken_bed(r.get('bed'))}: {str(r.get('action')).replace(' — ', ', ')}"
               for r in high[:3]]
        return (f"{n} outstanding action items across {len(beds)} beds, {len(high)} high priority."
                + (" Top: " + "; ".join(top) + "." if top else "")
                + " Ask about a bed for its actions.")
    if name == "cohort_patterns":
        top = rows[:3]
        sexes = {"m": "men", "f": "women"}
        said = [f"{spoken_event(r.get('event'))} in "
                f"{sexes.get(str(r.get('gender') or '').lower(), 'patients')} aged "
                f"{str(r.get('age_band')).replace('65+', '65 and over').replace('<40', 'under 40')}"
                f", {r.get('n')} event{'s' if r.get('n') != 1 else ''}" for r in top]
        return f"Most common patterns: {'; '.join(said)}."
    if any(r.get("event") or r.get("reported_event") or r.get("event_type") for r in rows):
        return _spoken_event_summary(rows)
    shown = [_row_to_sentence(r) for r in rows[:3]]
    more = f" And {n - 3} more; ask for details." if n > 3 else ""
    return f"{n} records. " + " ".join(f"{s.rstrip('.')}." for s in shown) + more


_SEVERITY = ("Critical", "High", "Medium", "Low")


def _answer_most_severe(rows: list[dict]) -> str:
    """'Which bed has the most severe alarm?': the beds in the top open tier, not the whole list."""
    if not rows:
        return "No unacknowledged alarms."
    level = next((lv for lv in _SEVERITY if any(r.get("criticality") == lv for r in rows)),
                 rows[0].get("criticality"))
    top = [r for r in rows if r.get("criticality") == level]
    counts: dict[tuple, int] = {}
    for r in top:
        key = (spoken_bed(r.get("bed")), spoken_event(r.get("event")))
        counts[key] = counts.get(key, 0) + 1
    items = [f"{bed}, {event}{_times(c)}" for (bed, event), c in counts.items()]
    rest = len(rows) - len(top)
    return (f"Most severe: {str(level).lower()}, on {_spoken_list(items)}."
            + (f" {rest} less severe unacknowledged event{'s' if rest != 1 else ''}." if rest else ""))


def _spoken_critical_by_bed(rows: list[dict], hours: int | None) -> str:
    """High/critical events, bed by bed, worst bed first:
    '5 high or critical events in the last 24 hours, on 2 beds. Bed 2: critical ventricular
    tachycardia twice. Bed 8: high PAC 3 times.'"""
    window = f" in the last {hours} hours" if hours else ""
    if not rows:
        return f"No high or critical events{window}."
    by_bed: dict[str, dict[tuple, int]] = {}
    for r in rows:  # newest first, as the template returns them
        key = (str(r.get("criticality") or ""), spoken_event(r.get("event")))
        bed = by_bed.setdefault(spoken_bed(r.get("bed")), {})
        bed[key] = bed.get(key, 0) + 1

    def rank(level: str) -> int:
        return _SEVERITY.index(level) if level in _SEVERITY else len(_SEVERITY)

    def bed_no(bed: str):
        m = re.search(r"\d+", bed)
        return int(m.group()) if m else 10**6

    order = sorted(by_bed, key=lambda b: (min(rank(lv) for lv, _ in by_bed[b]), bed_no(b)))
    parts = [f"{len(rows)} high or critical events{window}, on {len(by_bed)} "
             f"bed{'s' if len(by_bed) != 1 else ''}."]
    for bed in order:
        kinds = sorted(by_bed[bed].items(), key=lambda kv: rank(kv[0][0]))
        said = ", ".join(f"{lv.lower()} {ev}{_times(c)}" for (lv, ev), c in kinds)
        parts.append(f"{bed.capitalize()}: {said}.")
    return " ".join(parts)


def _spoken_event_summary(rows: list[dict]) -> str:
    """Every event type, counted — never truncated (a bed's VF was dropped by "first three").

    'Bed 2, patient PT998224: 4 events — ventricular tachycardia twice, ventricular fibrillation,
    atrial fibrillation. 4 unacknowledged.'
    """
    def _event(r):
        return r.get("event") or r.get("reported_event") or r.get("event_type")

    who = []
    beds = {r.get("bed") for r in rows if r.get("bed")}
    patients = {r.get("patient") for r in rows if r.get("patient")}
    if len(beds) == 1:
        who.append(spoken_bed(next(iter(beds))).capitalize())
    if len(patients) == 1:
        who.append(f"patient {next(iter(patients))}")
    counts: dict[str, int] = {}
    for r in rows:  # newest first, as the templates return them
        name = spoken_event(_event(r))
        counts[name] = counts.get(name, 0) + 1
    kinds = ", ".join(f"{name}{_times(c)}" for name, c in counts.items())
    head = f"{', '.join(who)}: " if who else ""
    text = f"{head}{len(rows)} events — {kinds}."
    if any("status" in r for r in rows):
        # Same rule as the Events tab and `cli.events`: pending = alerted and not closed. Counting
        # every open row said "5 unacknowledged" for a bed with one alert and four withheld events.
        pending = sum(1 for r in rows if is_pending(r))
        text += f" {pending} unacknowledged." if pending else " Nothing awaiting acknowledgment."
    if len(beds) > 1 and len(patients) == 1:
        text += f" Beds: {', '.join(spoken_bed(b) for b in sorted(beds))}."
    return text


def _answer_event_history(rows: list[dict], params: dict, *, spoken: bool = False) -> str:
    """An event list that states its own filter, so it can be checked row for row on the Events
    tab (or with `cli.events`). Text lists every event with a short id; spoken counts them."""
    desc = describe_event_history(params)
    if not rows:
        return f"No {desc}."
    n = len(rows)
    one = re.sub(r"\bevents\b", "event", desc, count=1) if n == 1 else desc  # not "1 events"
    if spoken:  # "bed 6", not "bed Unit1-Bed06"
        one = re.sub(r"bed (Unit\d+-Bed\d+)", lambda m: spoken_bed(m.group(1)), one)
    # "How many patients…" was answered with an event count: one patient can have several events
    # (PT998224: VF and VT), so a listing not already scoped to a patient also says how many.
    if n > 1 and not params.get("patient_id"):
        patients = len({r["patient"] for r in rows if r.get("patient")})
        if patients:
            one += f" for {patients} patient{'' if patients == 1 else 's'}"
    if spoken:
        kinds: dict[str, int] = {}
        for r in rows:
            k = spoken_event(r.get("event"))
            kinds[k] = kinds.get(k, 0) + 1
        text = f"{n} {one}: " + ", ".join(f"{k}{_times(c)}" for k, c in kinds.items()) + "."
        beds = sorted({r["bed"] for r in rows if r.get("bed")})
        if len(beds) > 1 and not params.get("bed"):
            text += f" Beds: {', '.join(spoken_bed(b) for b in beds)}."
        if (params.get("status") or "any") == "any":
            pending = sum(1 for r in rows if r.get("status") == "pending")
            text += f" {pending} pending." if pending else " None pending."
        return text
    lines = []
    for r in rows:
        parts = [_fmt_value("ts", r["ts"]) if r.get("ts") is not None else "time unknown"]
        if not params.get("bed"):
            parts.append(r.get("bed") or "bed unknown")
        if not params.get("patient_id"):
            parts.append(r.get("patient") or "patient unknown")
        event = _fmt_value("event", r.get("event") or "event")
        if r.get("confidence") is not None:
            event += f" {r['confidence']:.0%}"
        parts += [event, r.get("criticality") or "criticality unknown", r.get("status") or "?",
                  f"id {str(r.get('event_id') or '?')[:8]}"]
        lines.append("- " + " · ".join(parts))
    return f"{n} {one}:\n" + "\n".join(lines)


def _answer_operational(rows: list[dict] | None, name: str | None = None) -> str:
    """Deterministic, readable answer for a structured graph result (no LLM).

    Operational queries return exact rows, so we render them directly — accurate every time, no model
    latency or hallucination. A single row is one natural sentence. For multiple rows we hoist the
    fields shared by every row (patient/bed/unit/…) into a one-line header, then list each record on
    its own line ('- …') with only its distinguishing fields — compact, no repeated boilerplate, and
    the newlines give TTS a pause between records.
    """
    rows = [{k: v for k, v in r.items() if v is not None} for r in (rows or [])]
    if not rows:
        return _EMPTY_ANSWERS.get(name or "", "No matching records.")
    if len(rows) == 1:
        return _row_to_sentence(rows[0])
    # Context fields (patient/bed/unit) identical across every row are hoisted into the header.
    shared = {
        k: rows[0][k]
        for k in _CONTEXT_KEYS
        if k in rows[0] and all(r.get(k) == rows[0].get(k) for r in rows)
    }
    per_row = [_row_to_sentence({k: v for k, v in r.items() if k not in shared}) for r in rows]
    context = _describe_context(shared)
    header = f"{len(rows)} records for {context}:" if context else f"{len(rows)} matching records:"
    return header + "\n" + "\n".join(f"- {s}" for s in per_row)


def _describe_llm_error(exc: Exception) -> str:
    """One-line, prompt-free summary of a provider failure: type, HTTP status, short message.

    Never the prompt: provider errors (auth, rate limit, timeout, connection refused) describe the
    transport, and the message is truncated and passed through the redacting logger regardless.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    msg = " ".join(str(exc).split())[:200]
    return f"{type(exc).__name__}" + (f" [{status}]" if status else "") + (f": {msg}" if msg else "")


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
        min_relevance: float | None = None,
        llm_router: bool | None = None,
    ) -> None:
        self.working = working
        self.hybrid = hybrid
        self.episodic = episodic
        self.llm = llm
        self.driver = driver
        self.guardrails = guardrails or Guardrails()
        self.llm_retries = llm_retries
        self.episodic_recall = episodic_recall  # gate recalled "past interactions" in free-text answers
        # Semantic arm of the relevance gate; see Config.kb_min_relevance for the calibration.
        self.min_relevance = DEFAULT.kb_min_relevance if min_relevance is None else min_relevance
        # LLM fallback routing for operational questions the regexes miss (kb/graph/llm_router.py).
        self.llm_router = DEFAULT.kb_llm_router if llm_router is None else llm_router

    def _model_input(self, parts: list[tuple[str, bool]]) -> str:
        """What the model received (de-identified), for the audit; '' if de-id fails here."""
        try:
            if hasattr(self.llm, "deidentify_parts"):
                return self.llm.deidentify_parts(parts)
            return self.llm.deidentifier.deidentify("".join(t for t, _ in parts))
        except Exception:  # noqa: BLE001 - audit text only; the answer path already failed closed
            return ""

    def _generate(self, prompt: "str | list[tuple[str, bool]]") -> tuple[str, bool]:
        """Generate with one retry on transient failure; return (text, failed). Never raises.

        `prompt` is plain text (fully de-identified) or `(text, sensitive)` parts (only the
        sensitive parts de-identified) when the LLM supports it.
        """
        from common.deid import DeidError  # noqa: PLC0415

        if isinstance(prompt, list):
            if hasattr(self.llm, "generate_parts"):
                call = partial(self.llm.generate_parts, prompt)
            else:
                call = partial(self.llm.generate, "".join(t for t, _ in prompt))
        else:
            call = partial(self.llm.generate, prompt)
        attempts = self.llm_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return call(), False
            except DeidError:
                return "I couldn't safely process that request.", True  # fail closed
            except Exception as exc:  # noqa: BLE001 - timeout/rate-limit/unreachable -> retry then fall back
                _log.warning("LLM generate failed (attempt %d/%d): %s", attempt, attempts,
                             _describe_llm_error(exc))
        _log.error("LLM unavailable after %d attempt(s); returning fallback answer", attempts)
        return _LLM_FALLBACK, True

    def _protocol_answer(self, question: str, state, spoken: bool):
        """Structured steps, else the document's sections, for a protocol question naming a rhythm
        (or following one up: "what about its long-term management?", "the rest"). None = not one."""
        q = normalize_spoken_query(question)
        etype, scope = event_type_from_text(q), q
        if etype is None:
            prev = next((t.text for t in reversed(state.turns[:-1]) if t.role == "clinician"), "")
            if names_protocol(prev) and (names_protocol(q) or _REFERS_BACK.search(q)
                                         or protocol_answer.wants_more(q)):
                scope = f"{normalize_spoken_query(prev)} {q}"  # "go on" keeps the RVR scope
                etype = event_type_from_text(scope)
        elif not names_protocol(q):
            return None
        if etype is None or etype == NORMAL_SINUS:
            return None
        try:
            rows = run_template(self.driver, "care_protocol_steps", event_type=etype) \
                if self.driver is not None else []
        except Exception:  # noqa: BLE001 - graph down: the documents can still answer
            rows = []
        if rows and rows[0].get("steps") and protocol_answer.steps_apply(rows[0], scope):
            print(f"[orchestrator] protocol for {etype}: structured steps", flush=True)
            return protocol_answer.answer_from_steps(rows[0], etype, spoken=spoken, question=q)
        try:
            hits = self.hybrid.vector.search(f"{spoken_event(etype)} protocol {q}", k=20,
                                             rerank=False)
        except Exception:  # noqa: BLE001 - no vector store: fall back to the normal path
            return None
        answer = protocol_answer.answer_from_sections(hits, etype, spoken=spoken, question=q)
        if answer is not None:
            print(f"[orchestrator] protocol for {etype}: document sections {answer.citations}",
                  flush=True)
        return answer

    def handle_turn(
        self, session_id: str, user_text: str, *, patient_ref: str | None = None,
        now: float | None = None, spoken: bool = False,
    ) -> TurnResult:
        """One clinician turn. `spoken=True` (a voice turn) shortens multi-row graph answers."""
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

        # --- courtesy: "thank you" / "goodbye" -> a short reply, no KB/LLM ---
        courtesy = None
        if _COURTESY.match(user_text):  # "ok"/"great" alone are not pleasantries: they fall through
            if _FAREWELL.search(user_text):
                courtesy = _GOODBYE_REPLY
            elif _THANKS.search(user_text):
                courtesy = _YOU_ARE_WELCOME
        if courtesy:
            with tracer.span("courtesy"):
                self.working.append_turn(
                    session_id, ChatTurn(role="assistant", text=courtesy, timestamp=now + 0.001)
                )
            return TurnResult(answer=courtesy, mode="courtesy", trace=tracer.as_dicts())

        # --- "what does a MEWS of N mean?" -> the scorer's band + the document's escalation rule ---
        if _MEANING.search(user_text) and not names_bed_label(user_text) \
                and (m := _MEWS_SCORE.search(user_text)) and int(m.group(1)) <= 20:
            escalation = next((c["text"] for c in self.hybrid.vector.store.chunks_for_doc(_MEWS_DOC)
                               if str(c.get("source", "")).endswith("#Escalation thresholds")), None)
            if escalation:  # the chunk opens with its heading line
                escalation = escalation.split("\n", 1)[-1].replace("\n", " ")
            answer_text = _answer_mews_meaning(int(m.group(1)), escalation)
            self.working.append_turn(
                session_id, ChatTurn(role="assistant", text=answer_text, timestamp=now + 0.001))
            return TurnResult(answer=answer_text, mode="kb",
                              citations=[f"{_MEWS_DOC}#Escalation thresholds"] if escalation else [],
                              trace=tracer.as_dicts())

        # --- intent: operational template vs hybrid KB ---
        operational_rows: list[dict] | None = None
        with tracer.span("retrieve") as sp:
            intent = match_intent(user_text, now=now, patient_ref=state.patient_ref,
                                  event_ref=state.event_ref)
            # "Protocol for this event" / a bed's protocol with no structured steps: a document
            # question about that event's rhythm, asked in those words.
            kb_question = user_text
            if intent is None and asks_event_protocol(user_text, state.event_ref):
                rows = run_template(self.driver, "event_summary", event_uuid=state.event_ref)
                kb_question = _protocol_question(rows[0].get("event_type") if rows else None) \
                    or user_text
            elif intent and intent[0] == "protocol_for_bed_last_event":
                rows = run_template(self.driver, *intent[:1], **intent[1])
                if not rows or not rows[0].get("steps"):
                    last = run_template(self.driver, "bed_last_event", bed=intent[1]["bed"])
                    question = _protocol_question(last[0].get("event_type") if last else None)
                    if question:
                        kb_question, intent = question, None
            rewritten = kb_question != user_text
            if rewritten:
                print(f"[orchestrator] protocol question for the event's rhythm: {kb_question!r}",
                      flush=True)
            decline_hint = None if intent is not None or rewritten else (
                _EVENT_LISTING_HINT if looks_like_event_listing(user_text)
                else _BED_HINT if names_unresolved_bed(user_text) else None)
            listing_declined = decline_hint is not None
            if listing_declined:
                print("[orchestrator] unresolvable target (event listing or bed): declined with a "
                      "hint", flush=True)
            emergency = is_emergency(user_text)
            if intent is None and self.llm_router and not listing_declined and not rewritten \
                    and not emergency:
                # The regexes only fire on phrasings someone anticipated. When one misses a question
                # that is plainly about a bed/patient/event, ask the model which template was meant
                # — it picks a name from a fixed menu and never supplies an identity. Gated on
                # `looks_operational` inside, so document questions never pay for this.
                intent = llm_route(user_text, self.llm, patient_ref=state.patient_ref,
                                   event_ref=state.event_ref, now=now)
                if intent is not None and state.event_ref and intent[0] in _CALLED_EVENT_TWIN:
                    intent = (_CALLED_EVENT_TWIN[intent[0]], {"event_uuid": state.event_ref})
                if intent is not None:
                    print(f"[orchestrator] llm-routed to template {intent[0]!r}", flush=True)
                    sp.attributes["llm_routed"] = intent[0]
            proto = None if intent is not None or listing_declined or emergency \
                else self._protocol_answer(kb_question, state, spoken)
            if proto is not None:
                sp.attributes["mode"] = "protocol"
                self.working.append_turn(session_id, ChatTurn(
                    role="assistant", text=proto.text, timestamp=now + 0.001, mode="kb"))
                return TurnResult(answer=proto.text, mode="kb", citations=proto.citations,
                                  trace=tracer.as_dicts())
            if intent is None and not rewritten and not listing_declined and not emergency \
                    and is_patient_specific(user_text):
                decline_hint, listing_declined = _PATIENT_HINT, True
                print("[orchestrator] patient-specific question with no graph query: hint, not "
                      "free generation", flush=True)
            if intent:
                name, params = intent
                operational_rows = run_template(self.driver, name, **params)
                shown = {k: v for k, v in params.items() if k != "since"}
                print(f"[orchestrator] graph query {name} {shown} -> {len(operational_rows)} row(s)",
                      flush=True)
                kb_context = _render_operational(name, operational_rows)
                citations = [f"graph:{name}"]
                mode, declined = "operational", False
            elif listing_declined:
                mode, declined, kb_context, citations = "operational", False, "", []
            else:
                mode = "hybrid"
                result = self.hybrid.retrieve(kb_question, mode="hybrid")
                # Archived event reports are patient records: they belong to graph answers, not to
                # free-text ones (the model quoted their IDs and invented patients from them).
                kept = [p for p in result.passages if not str(p.source).startswith("report:")]
                if len(kept) != len(result.passages):
                    result = result.model_copy(update={"passages": kept})
                kb_parts, citations = _render_block_parts(result)
                kb_context = "".join(t for t, _ in kb_parts)
                overlap = _best_overlap(kb_question, result.passages) if result.passages else 0.0
                top_score = result.passages[0].score if result.passages else 0.0
                # Two independent ways a passage can vouch for itself: it *means* the same thing
                # (semantic similarity, which survives rewording) or it *says* the same words. Either
                # is enough. The lexical arm alone used to decide, which is why a correctly retrieved
                # passage phrased differently from the question was refused.
                declined = not is_relevant(result, overlap, top_score,
                                           min_relevance=self.min_relevance)
                missing = [] if result.relationships or emergency else _missing_topic(
                    kb_question, result.passages)
                if not declined and missing and top_score < _TOPIC_SIMILARITY:
                    declined = True
                    print(f"[orchestrator] declined: no retrieved passage mentions "
                          f"{', '.join(missing)} (similarity {top_score:.2f} < "
                          f"{_TOPIC_SIMILARITY:.2f})", flush=True)
                sp.attributes.update(top_score=round(top_score, 3), overlap=round(overlap, 3))
                if declined and not missing:
                    # "I don't have information on that" is indistinguishable from a broken KB
                    # unless the turn says which of the three ways it got there. Logged, traced,
                    # and carried on the result so a caller can surface it.
                    decline_detail = explain_decline(result, overlap, top_score,
                                                     min_relevance=self.min_relevance)
                    sp.attributes["decline_reason"] = decline_detail
                    print(f"[orchestrator] declined: {decline_detail}", flush=True)
            sp.attributes.update(mode=mode, declined=declined)

        # --- output policy: answer / decline / escalate ---
        action = self.guardrails.decide_output(user_text, declined=declined)
        escalated = action == "escalate"
        model_input = ""  # what the model received, de-identified (audit); "" when no model is called
        if action == "escalate":
            answer_text = self.guardrails.refusal_for_emergency
        elif action == "decline":
            answer_text = _DECLINE
        elif mode == "operational" and listing_declined:
            answer_text = decline_hint
        elif mode == "operational" and name == EVENT_HISTORY:
            answer_text = _answer_event_history(operational_rows, params, spoken=spoken)
            model_input = self._model_input([(kb_context[:_AUDIT_MAX_CHARS], True)])
        elif mode == "operational":
            # Structured graph result -> deterministic, crisp answer. No LLM is called, so NO prompt
            # is built: conversation history and recalled past interactions never touch an operational
            # answer. (They are only useful for free-text follow-ups, below.)
            focus = params.get("focus") or ("mews" if name.startswith("mews_") else None)
            if name == "patient_clinical_history":
                answer_text = _answer_clinical_history(operational_rows, focus)
            elif name == "event_summary":
                answer_text = (_answer_event_summary(operational_rows[0], focus) if operational_rows
                               else "I couldn't find that event.")
            elif name == "action_items_for_event":
                answer_text = _answer_action_items(operational_rows)
            elif name == "patients_by_criticality":
                answer_text = _answer_patient_ranking(operational_rows, order=params.get("order"),
                                                      spoken=spoken)
            elif name == "vital_readings_for_event":
                answer_text = (_answer_vital_readings(operational_rows[0], focus, params.get("n", 3))
                               if operational_rows else "I couldn't find that event.")
            elif name == "event_count_since":
                answer_text = _answer_event_count(operational_rows, params.get("window"))
            elif name == "worklist" and params.get("most_severe"):
                answer_text = _answer_most_severe(operational_rows)
            elif name == "critical_events_since" and spoken:
                answer_text = _spoken_critical_by_bed(operational_rows, params.get("hours"))
            elif spoken and operational_rows and len(operational_rows) == 1 \
                    and name in _RECORD_TEMPLATES:
                answer_text = _spoken_record(operational_rows[0], focus)
            else:
                answer_text = (_spoken_summary(operational_rows, name)
                               if spoken and operational_rows and len(operational_rows) > 1
                               else _answer_operational(operational_rows, name))
            # The rows that informed it (audit only): a de-id failure must not lose this answer.
            model_input = self._model_input([(kb_context[:_AUDIT_MAX_CHARS], True)])
        else:
            # Free-text/hybrid: the LLM needs conversation history (multi-turn follow-ups). Built ONLY
            # here — only when the model is actually invoked. Recalled "past interactions" are added
            # only when `episodic_recall` is on (off by default: grounded in live KB + this chat only).
            with tracer.span("build_context"):
                # Only a question that refers back ("what about its dose?") gets the history, and
                # never the graph answers in it: the 3B model answered a VT protocol question with
                # the worklist's beds, and a PVC question with the previous VT answer.
                history = "\n".join(f"{t.role}: {t.text}" for t in state.turns[-_MAX_HISTORY:-1]
                                     if t.mode != "operational") \
                    if _REFERS_BACK.search(user_text) else ""
                # (text, sensitive): only parts that can carry PHI are de-identified (see
                # DeidentifyingLLM.deidentify_parts); the joined text is unchanged from before.
                parts: list[tuple[str, bool]] = [
                    (_ANSWER_INSTRUCTIONS + (_SPOKEN_INSTRUCTIONS if spoken else ""), False),
                    ("\n\n## Conversation so far\n", False), (history or "(none)", True),
                ]
                if self.episodic_recall:
                    episodes = self.episodic.recall(user_text, k=3, patient_ref=state.patient_ref)
                    episodic_ctx = "\n".join(f"- {e.text}" for e in episodes) or "(none)"
                    parts += [("\n\n## Relevant past interactions\n", False), (episodic_ctx, True)]
                parts += [("\n\n", False), *kb_parts, ("\n\nQuestion: ", False),
                          (kb_question, True), ("\nAnswer:", False)]
                prompt = "".join(t for t, _ in parts)
            with tracer.span("generate") as sp:
                answer_text, failed = self._generate(parts)  # de-id'd + retried inside
                if spoken:  # "(P1)", "[R8]": citation labels are for the screen, not to be read out
                    answer_text = re.sub(r"\s*[\[(][PR]\d+\b[^\])]*[\])]", "", answer_text).strip()
                sp.attributes["llm_failed"] = failed
                # Output guardrail: a patient/bed/ward/event the model was never given is invented
                # or misattributed — on a clinical relay worse than no answer. Seen live: "Patient
                # ID 1234, bed Bayside 3" with an empty patient graph.
                foreign = ungrounded_identifiers(answer_text, prompt, state.patient_ref)
                if foreign:
                    print(f"[orchestrator] blocked answer naming identifier(s) not in context: "
                          f"{', '.join(foreign)}", flush=True)
                    sp.attributes["blocked_patient_refs"] = foreign
                    answer_text = UNGROUNDED_ANSWER
                # A de-id placeholder in the answer means the model echoed a scrubbed span
                # ("<PERSON> is not associated…"); reading it aloud is meaningless and confusing.
                if not foreign and _PLACEHOLDER.search(answer_text):
                    print(f"[orchestrator] replaced answer echoing a de-id placeholder: "
                          f"{_PLACEHOLDER.findall(answer_text)}", flush=True)
                    sp.attributes["placeholder_echo"] = True
                    answer_text = _PLACEHOLDER_ANSWER
            model_input = self._model_input(parts)

        with tracer.span("persist"):
            self.working.append_turn(
                session_id, ChatTurn(role="assistant", text=answer_text, timestamp=now + 0.001,
                                     mode=mode)
            )
            if self.episodic_recall:  # only archive Q&A when it will actually be recalled
                self.episodic.add(
                    f"Q: {user_text}\nA: {answer_text}", session_id=session_id,
                    patient_ref=state.patient_ref, timestamp=now,
                )

        listed = mode == "operational" and not listing_declined and name == EVENT_HISTORY
        return TurnResult(
            answer=answer_text, citations=list(dict.fromkeys(citations)),
            mode=mode, declined=(action == "decline"), escalated=escalated,
            model_input=model_input, trace=tracer.as_dicts(),
            history_filter=history_filter_for(params) if listed and action == "answer" else None,
        )
