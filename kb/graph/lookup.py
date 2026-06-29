"""Operational lookup: natural-language intent -> parameterized template.

Tries a small library of intent rules first (the tested, parameterized templates). Free LLM
text-to-Cypher is a **read-only allowlisted fallback only** — any generated statement is run
through `run_read_only`, which refuses writes.
"""

from __future__ import annotations

import re

from common.event_types import event_type_from_text
from common.interfaces import LLMProvider

from .driver import GraphDriver
from .entities import condition_id
from .spoken import normalize_spoken_query
from .templates import run_template

_BED = re.compile(r"\bbed\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
_HOURS = re.compile(r"\b(\d+)\s*h(?:ours?)?\b", re.IGNORECASE)
_MINUTES = re.compile(r"\b(\d+)\s*m(?:in(?:utes?)?)?\b", re.IGNORECASE)
# Vitals question: any vital-sign term.
_VITALS = re.compile(
    r"\b(vitals?|vital\s+signs?|heart\s+rate|blood\s+pressure|"
    r"spo2|oxygen|saturation|respiratory\s+rate|resp\s+rate|temperature|hr|bp|rr)\b",
    re.IGNORECASE,
)
# "all/which patients with <event>" — list-by-event-type phrasing.
_ALL_PATIENTS = re.compile(
    r"\ball patients\b|\bwhich patients\b|\bpatients who\b|\bwho (?:has|have|had)\b|"
    r"\bshow (?:me )?all\b|\blist (?:all )?patients\b",
    re.IGNORECASE,
)
# The *analysis report* intent (T4). Word-boundary so "reported"/"reports" (the verb in
# "events reported on bed") does NOT trigger it — only the noun "report"/"analysis".
_REPORT = re.compile(r"\breport\b|\banalysis\b", re.IGNORECASE)
# "this/current/same patient" — scopes a query to the session's patient (outbound call).
_THIS_PATIENT = re.compile(r"\b(this|current|same) patient(?:'s|s)?\b", re.IGNORECASE)


def match_intent(query: str, *, now: float, patient_ref: str | None = None) -> tuple[str, dict] | None:
    """Map a natural-language query to (template_name, params), or None.

    Covers the operational matrix (T1–T10 + traversals). `patient_ref` scopes the session-relative
    vitals intent ("the event" = that patient's latest); bed/event-type-scoped intents read their
    target from the query text, so they work on the inbound text path too. Rules are ordered most-
    to least-specific (first match wins). The query is first normalized for spoken/STT phrasing
    (spelled acronyms, number-word bed labels) so voice and typed queries route identically.
    """
    query = normalize_spoken_query(query)
    q = query.lower()
    bed = m.group(1) if (m := _BED.search(query)) else None
    etype = event_type_from_text(q)

    # T9 — ECG strips for the last event of a type ("ECG strips for the last AFib event").
    if etype and ("ecg" in q or "strip" in q):
        return "ecg_strips_last_event_of_type", {"event_type": etype}

    # T10 — HR/BP trend for the last event of a type ("HR and BP trend for the last VT event").
    if etype and ("trend" in q or ("hr" in q and "bp" in q)):
        return "trend_last_event_of_type", {"event_type": etype}

    # List all patients who had an event of a type ("show all patients with an AFib event").
    if etype and _ALL_PATIENTS.search(q):
        return "patients_with_event_type", {"event_type": etype}

    # T5 — vitals at the event: bed-scoped if a bed is named, else the session patient's latest.
    if _VITALS.search(q) and "trend" not in q:
        if bed:
            return "vitals_for_bed_last_event", {"bed": bed}
        if patient_ref:
            return "vitals_at_patient_last_event", {"patient_id": patient_ref}

    # "this/current patient" scoping (outbound call) — answer about the session's patient, not all.
    if patient_ref and _THIS_PATIENT.search(q):
        if "critical" in q and "event" in q:
            return "critical_events_for_patient", {"patient_id": patient_ref}
        if "event" in q:  # "events for this patient", "this patient's events"
            return "events_for_patient", {"patient_id": patient_ref}

    # T1 — critical events in the last N hours.
    if "critical" in q and "event" in q:
        hours = int(m.group(1)) if (m := _HOURS.search(q)) else 24
        return "critical_events_since", {"since": now - hours * 3600}

    # T2 — positive (non-false-positive) events in the last x minutes.
    if ("positive" in q or "non-false" in q) and "event" in q:
        mins = int(m.group(1)) if (m := _MINUTES.search(q)) else 60
        return "positive_events_since", {"since": now - mins * 60}

    # T8 — demographic / co-morbidity / symptom patterns vs event type.
    if "pattern" in q:
        return "cohort_patterns", {}

    # T3 — event status on a bed. (Before T4 so "events reported on bed" stays a status query.)
    if ("status" in q or "what happened" in q) and bed:
        return "event_status_on_bed", {"bed": bed}

    # T7 — care protocol for a bed's last event. (Before T4 for the same reason.)
    if "protocol" in q and bed:
        return "protocol_for_bed_last_event", {"bed": bed}

    # T4 — event analysis report(s) for a bed (the noun "report"/"analysis").
    if _REPORT.search(q) and bed:
        return "reports_for_bed", {"bed": bed}

    # T6 — outstanding action items across patients.
    if "action item" in q or "outstanding" in q:
        return "outstanding_action_items", {}

    # Relationship lookup — co-morbidities of a named condition ("comorbidities of atrial fib").
    if "co-morbid" in q or "comorbid" in q:
        m = re.search(r"(?:of|with|for)\s+(.+?)\s*\??$", query, re.IGNORECASE)
        if m:
            return "comorbidity_neighborhood", {"condition_id": condition_id(m.group(1))}

    return None


def _looks_like_cypher(query: str) -> bool:
    return bool(re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH|CALL|RETURN|UNWIND)\b", query))


def lookup(driver: GraphDriver, query: str, *, now: float, llm: LLMProvider | None = None) -> dict:
    """Resolve a query to rows via a template (preferred) or a read-only Cypher fallback."""
    if _looks_like_cypher(query):
        # Raw Cypher path — run only if read-only (refuses writes).
        return {"mode": "raw_cypher", "rows": driver.run_read_only(query)}

    intent = match_intent(query, now=now)
    if intent:
        name, params = intent
        return {"mode": "template", "template": name, "rows": run_template(driver, name, **params)}

    if llm is not None:
        cypher = llm.generate(f"Translate to a single read-only Cypher query: {query}")
        return {"mode": "llm_cypher", "cypher": cypher, "rows": driver.run_read_only(cypher)}

    return {"mode": "declined", "rows": []}
