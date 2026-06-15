"""Operational lookup: natural-language intent -> parameterized template.

Tries a small library of intent rules first (the tested, parameterized templates). Free LLM
text-to-Cypher is a **read-only allowlisted fallback only** — any generated statement is run
through `run_read_only`, which refuses writes.
"""

from __future__ import annotations

import re

from common.interfaces import LLMProvider

from .driver import GraphDriver
from .entities import condition_id
from .templates import run_template

_BED = re.compile(r"\bbed\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
_HOURS = re.compile(r"\b(\d+)\s*h(?:ours?)?\b", re.IGNORECASE)
_MINUTES = re.compile(r"\b(\d+)\s*m(?:in(?:utes?)?)?\b", re.IGNORECASE)


def match_intent(query: str, *, now: float) -> tuple[str, dict] | None:
    """Map a natural-language query to (template_name, params), or None."""
    q = query.lower()

    if "critical" in q and "event" in q:
        hours = int(m.group(1)) if (m := _HOURS.search(q)) else 24
        return "critical_events_since", {"since": now - hours * 3600}

    if ("positive" in q or "non-false" in q) and "event" in q:
        mins = int(m.group(1)) if (m := _MINUTES.search(q)) else 60
        return "positive_events_since", {"since": now - mins * 60}

    if ("status" in q or "what happened" in q) and (m := _BED.search(query)):
        return "event_status_on_bed", {"bed": m.group(1)}

    if "action item" in q or "outstanding" in q:
        return "outstanding_action_items", {}

    if "protocol" in q and (m := _BED.search(query)):
        return "protocol_for_bed_last_event", {"bed": m.group(1)}

    if "co-morbid" in q or "comorbid" in q:
        # "comorbidities of atrial fibrillation"
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
