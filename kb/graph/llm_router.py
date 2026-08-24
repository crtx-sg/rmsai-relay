"""LLM fallback router: free-text question -> graph template, when the regexes in `lookup.py` miss.

`match_intent` is a first-match-wins list of regexes. It is fast, deterministic and auditable, and it
only fires on phrasings someone thought of in advance — so "how's the patient in bed three doing?"
falls through to document retrieval, which has nothing to say about a specific bed. This routes those
questions to the template they obviously meant.

Three properties make this safe enough to sit in front of Cypher:

* **The model picks a name, never a query.** Its choice is looked up in a fixed catalogue; anything
  not in the catalogue is discarded and the turn falls through to the hybrid retriever as before.
* **The model never supplies an identity.** `patient_id` and `event_uuid` come from session scope —
  the selected worklist row, the authenticated patient. A model that invents `PT9999` would
  otherwise pull another patient's chart, so those parameters are never taken from its output.
  Everything it *can* supply is enumerable and validated: an event type from the 16 known classes,
  a bed label matching the bed pattern, a time window as an integer.
* **It only runs on a miss, and only when the question smells operational** (`looks_operational`),
  so the common case — a clinical question answered from documents — pays nothing for it.
"""

from __future__ import annotations

import json
import re

from common.event_types import CLASS_NAMES

#: Routable templates: name -> (what it answers, the parameter the model must supply or None).
#: Deliberately a subset of `TEMPLATES` — every entry here is one a clinician asks in free text and
#: whose parameters can be validated. Identity-scoped entries take their id from the session.
CATALOGUE: dict[str, tuple[str, str | None]] = {
    "vitals_for_bed_last_event": ("vitals at the last event for a named bed", "bed"),
    "event_status_on_bed": ("status / what happened on a named bed", "bed"),
    "reports_for_bed": ("reports filed for a named bed", "bed"),
    "protocol_for_bed_last_event": ("care protocol for a named bed's last event", "bed"),
    "vitals_at_patient_last_event": ("vitals at this patient's most recent event", "session_patient"),
    "critical_events_for_patient": ("this patient's critical events", "session_patient"),
    "events_for_patient": ("this patient's event history", "session_patient"),
    "hr_trend_for_patient_last_event": ("HR trend at this patient's last event", "session_patient"),
    "vitals_at_selected_event": ("vitals at the selected worklist event", "session_event"),
    "critical_events_since": ("critical events in the last N hours", "hours"),
    "positive_events_since": ("non-false-positive events in the last N minutes", "minutes"),
    "ecg_strips_last_event_of_type": ("ECG strips for the last event of a rhythm type", "event_type"),
    "trend_last_event_of_type": ("HR/BP trend for the last event of a rhythm type", "event_type"),
    "patients_with_event_type": ("every patient who had a given rhythm type", "event_type"),
    "outstanding_action_items": ("open action items across the unit", None),
    "cohort_patterns": ("demographic / co-morbidity patterns vs event type", None),
}

#: A loose, high-recall gate: does this question sound like it is about *patients and monitoring*
#: rather than about clinical literature? Deliberately imprecise — precision is the model's job.
#: This exists purely to keep the router (and its seconds of latency) off the document path.
_OPERATIONAL = re.compile(
    r"\b(bed|patient|ward|unit|event|events|vitals?|vital signs|heart rate|hr|bp|blood pressure|"
    r"spo2|sats|mews|trend|trending|ecg|strip|rhythm|episode|alert|alarm|acknowledg\w*|status|"
    r"reported|outstanding|action items?|protocol for|last (hour|24|day)|today|tonight|"
    r"overnight|so far|right now|currently)\b",
    re.IGNORECASE,
)

_BED = re.compile(r"\b([A-Za-z][\w]*-Bed\d+|Bed\s*\d+)\b", re.IGNORECASE)


def _first_json_object(text: str) -> str | None:
    """The first balanced `{...}` span in `text`, or None.

    A non-greedy regex stops at the first closing brace, which truncates exactly the replies that
    matter — `{"name": ..., "params": {...}}` has a nested object — so the braces are counted.
    Braces inside string literals are skipped so a bed label containing one cannot unbalance it.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def looks_operational(query: str) -> bool:
    """Whether a question is about patients/monitoring at all — the cheap gate before any LLM call."""
    return bool(_OPERATIONAL.search(query or ""))


def build_prompt(query: str, *, has_patient: bool, has_event: bool) -> str:
    """The routing prompt: a fixed menu, one JSON object out, and an explicit escape hatch."""
    lines = []
    for name, (what, param) in CATALOGUE.items():
        if param == "session_patient" and not has_patient:
            continue  # no patient in scope: offering it invites a hallucinated id
        if param == "session_event" and not has_event:
            continue
        need = {
            "bed": ' needs "bed" (e.g. "Unit1-Bed01")',
            "hours": ' needs "hours" (integer)',
            "minutes": ' needs "minutes" (integer)',
            "event_type": f' needs "event_type" (one of: {", ".join(sorted(CLASS_NAMES))})',
        }.get(param, "")
        lines.append(f"- {name}: {what}{need}")
    menu = "\n".join(lines)
    return (
        "Pick the ONE query that answers the clinician's question, or none.\n\n"
        f"Queries:\n{menu}\n\n"
        'Reply with ONLY a JSON object: {"name": "<query name>", "params": {...}} — or '
        '{"name": null} if no query above fits. Never invent a patient id or an event id.\n\n'
        f"Question: {query}\nJSON:"
    )


def parse_choice(raw: str) -> tuple[str, dict] | None:
    """Pull `(name, params)` out of a model reply. Returns None for 'no match' or unusable output.

    Small local models wrap JSON in prose or code fences and sometimes emit nothing usable, so this
    takes the first object-looking span and tolerates junk around it. Unparseable output is a
    no-match, not an error — the caller simply falls through to the hybrid retriever.
    """
    span = _first_json_object(raw or "")
    if not span:
        return None
    try:
        data = json.loads(span)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not name or not isinstance(name, str):
        return None
    params = data.get("params")
    return name, (params if isinstance(params, dict) else {})


def resolve(choice: tuple[str, dict] | None, *, patient_ref: str | None = None,
            event_ref: str | None = None, now: float) -> tuple[str, dict] | None:
    """Validate a model's choice into `(template, params)` the graph can actually run, or None.

    This is the layer that makes the router safe: an unknown template name, a rhythm outside the
    known classes, a bed that isn't shaped like a bed, or a missing session identity all resolve to
    None rather than to a query. Identity parameters are filled from scope and never from the model.
    """
    if choice is None:
        return None
    name, params = choice
    entry = CATALOGUE.get(name)
    if entry is None:
        return None
    _, needs = entry

    if needs is None:
        return name, {}
    if needs == "session_patient":
        return (name, {"patient_id": patient_ref}) if patient_ref else None
    if needs == "session_event":
        return (name, {"event_uuid": event_ref}) if event_ref else None
    if needs == "bed":
        bed = str(params.get("bed", "")).strip()
        return (name, {"bed": bed}) if _BED.fullmatch(bed) else None
    if needs == "event_type":
        etype = str(params.get("event_type", "")).strip().upper().replace(" ", "_")
        return (name, {"event_type": etype}) if etype in CLASS_NAMES else None
    if needs in ("hours", "minutes"):
        try:
            amount = int(params.get(needs))
        except (TypeError, ValueError):
            return None
        if not 1 <= amount <= 10000:
            return None
        seconds = amount * (3600 if needs == "hours" else 60)
        return name, {"since": now - seconds}
    return None


def route(query: str, llm, *, patient_ref: str | None = None, event_ref: str | None = None,
          now: float) -> tuple[str, dict] | None:
    """Free-text question -> `(template, params)`, or None to fall through. Never raises.

    An LLM that is slow, down, or babbling must cost this turn nothing but time — the hybrid
    retriever behind it is a complete answer path on its own.
    """
    if not looks_operational(query):
        return None
    try:
        raw = llm.generate(build_prompt(query, has_patient=bool(patient_ref),
                                        has_event=bool(event_ref)))
    except Exception:  # noqa: BLE001 - routing is an optimisation; never break the turn
        return None
    return resolve(parse_choice(raw), patient_ref=patient_ref, event_ref=event_ref, now=now)
