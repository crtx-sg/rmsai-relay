"""Operational lookup: natural-language intent -> parameterized template.

Tries a small library of intent rules first (the tested, parameterized templates). Raw Cypher
(the `cli.graph lookup` dev path) runs through `run_read_only`, which refuses writes.
"""

from __future__ import annotations

import re

from common.event_types import event_type_from_text

from .driver import GraphDriver
from .entities import condition_id
from .spoken import normalize_spoken_query
from .templates import EVENT_HISTORY, run_template

_BED = re.compile(r"\bbed\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
# A full label typed as-is ("events on Unit1-Bed02"): no "bed " before it, so _BED misses it.
_BED_LABEL = re.compile(r"\bUnit\d+-Bed\d+\b", re.IGNORECASE)
_HOURS = re.compile(r"\b(\d+)\s*h(?:ours?)?\b", re.IGNORECASE)
_MINUTES = re.compile(r"\b(\d+)\s*m(?:in(?:utes?)?)?\b", re.IGNORECASE)
# Vitals question: any vital-sign term.
# Which single vital a question asks about ("what was the heart rate?"): the spoken answer then gives
# that one, not all six. Checked in order; "vitals" alone matches none of these.
_VITAL_FOCUS = (
    ("hr", re.compile(r"\b(heart\s+rate|hr|pulse)\b", re.IGNORECASE)),
    ("bp", re.compile(r"\b(blood\s+pressure|bp|systolic|diastolic)\b", re.IGNORECASE)),
    ("spo2", re.compile(r"\b(spo2|oxygen|saturation|sats)\b", re.IGNORECASE)),
    ("rr", re.compile(r"\b(respiratory\s+rate|resp\s+rate|breathing|rr)\b", re.IGNORECASE)),
    ("temp", re.compile(r"\b(temperature|temp|fever)\b", re.IGNORECASE)),
)
# The patient's clinical background (graph: diagnoses, medications, surgeries, symptoms), as opposed
# to their monitor *events*. "past medical history" used to list events; "what medications is the
# patient on" matched nothing at all.
_CLINICAL_HISTORY = re.compile(
    r"\b(medical\s+history|past\s+history|clinical\s+history|background|medications?|meds|"
    r"drugs?|prescri\w*|diagnos[ie]s|comorbidit\w*|co-morbidit\w*|conditions?|surger(?:y|ies)|"
    r"operations?|procedures?|allerg\w*|symptoms?|history|how\s+old|age|gender|sex)\b",
    re.IGNORECASE)
_HISTORY_FOCUS = (
    ("medications", re.compile(r"\b(medications?|meds|drugs?|prescri\w*)\b", re.IGNORECASE)),
    ("conditions", re.compile(r"\b(diagnos[ie]s|comorbidit\w*|co-morbidit\w*|conditions?)\b",
                              re.IGNORECASE)),
    ("surgeries", re.compile(r"\b(surger(?:y|ies)|operations?|procedures?)\b", re.IGNORECASE)),
    ("symptoms", re.compile(r"\bsymptoms?\b", re.IGNORECASE)),
    ("allergies", re.compile(r"\ballerg\w*", re.IGNORECASE)),
    ("demographics", re.compile(r"\b(how\s+old|age|gender|sex)\b", re.IGNORECASE)),
)
# Questions about the called/selected event itself (answered from that event, by uuid). Each maps to
# the part of `event_summary` the answer leads with.
_EVENT_FOCUS = (
    ("false_positive", re.compile(r"\bfalse[\s-]+(?:positive|alarm)\b|"
                                  r"\b(?:real|genuine|true)\s+(?:event|alarm|alert|arrhythmia)\b",
                                  re.IGNORECASE)),
    ("why", re.compile(r"\bwhy\b.*\b(?:alert\w*|call\w*|flag\w*|shown|notif\w*|paged|rang|ring\w*|"
                       r"contacted)\b|\breason\s+for\s+(?:the\s+)?(?:alert|call|alarm)\b",
                       re.IGNORECASE)),
    ("confidence", re.compile(r"\b(?:confiden\w*|how\s+sure|certain\w*)\b", re.IGNORECASE)),
    # The event's time needs the event as its subject: "when is the next dose", "what time is it"
    # and "when was the patient admitted" are not about it.
    ("time", re.compile(
        r"\b(?:when|what\s+time|how\s+long\s+ago)\s+(?:did|was|were)\s+(?:it|(?:this|the|that)\s+"
        r"(?:event|alarm|alert|arrhythmia|rhythm|episode))\b", re.IGNORECASE)),
    ("criticality", re.compile(r"\b(?:criticality|severity|how\s+(?:serious|severe|critical|bad))\b",
                               re.IGNORECASE)),
    ("rhythm", re.compile(r"\b(?:rhythm|arrhythmia|what\s+(?:was|is)\s+(?:detected|the\s+(?:event|"
                          r"alert|alarm|diagnosis))|detected)\b", re.IGNORECASE)),
)
# "What should I do?", "next steps", "action items" for the called event.
_EVENT_ACTIONS = re.compile(
    r"\b(?:action\s+items?|next\s+steps?|what\s+(?:should|do|can|must)\s+(?:i|we)\s+do|"
    r"what\s+to\s+do|recommend\w*|actions?)\b", re.IGNORECASE)
# "protocol / how to manage / treat this (event|patient|rhythm)" — answered from the documents for
# the called event's rhythm (orchestrator rewrites the question with it).
_EVENT_PROTOCOL = re.compile(r"\b(?:protocol|guideline|manag\w*|treat\w*|handle|sop)\b",
                             re.IGNORECASE)
_THIS_TARGET = re.compile(
    r"\b(?:this|the|that|current)\s+(?:patient|event|alert|alarm|rhythm|arrhythmia|case)\b|"
    r"\b(?:his|her|him|he|she|patient's)\b|\bthe\s+model\b|\bnow\b", re.IGNORECASE)
# Kept out of the this-event rules: questions about a vital, the MEWS, the patient's history or a
# protocol have their own rules ("heart rate when the event was detected" is a vitals question),
# and "what is a false positive" asks for a definition, not about this event.
_NOT_EVENT_FOCUS = re.compile(
    r"\b(?:mews|news\s+(?:score|\d)|early\s+warning|warning\s+score)\b|"
    r"^\W*(?:what\s+(?:is|are)\s+(?:a|an)\b|what\s+does\b|define\b|explain\b)", re.IGNORECASE)
# "...for bradycardia", "...for hyperkalemia": a named topic, not this event's action items.
_FOR_OTHER = re.compile(
    r"\bfor\s+(?!this\b|the\s+patient\b|him\b|her\b|it\b|now\b|them\b)\w+|\bdos(?:e|es|ing)\b",
    re.IGNORECASE)
# The patient/event words that make a question patient-specific (no "now", "the model", "the alarm":
# "what does the alarm mean" and "what does the model do" are document questions).
_PATIENT_TARGET = re.compile(
    r"\b(?:this|that|current)\s+(?:patient|event|alert|alarm|rhythm|arrhythmia|case)\b|"
    r"\bthe\s+patient(?:'s)?\b|\b(?:his|her|him|he|she|patient's)\b", re.IGNORECASE)
_EMERGENCY_WORDS = re.compile(
    r"\b(?:coding|arrest\w*|code\s+blue|unresponsive|not\s+breathing|collaps\w*|no\s+pulse|"
    r"pulseless)\b", re.IGNORECASE)
_TODAY = re.compile(r"\b(?:today|this\s+(?:morning|afternoon|evening|shift)|so\s+far)\b",
                    re.IGNORECASE)
_PATIENT_WORD = re.compile(r"\b(patient|patient's|his|her|their|he|she|they)\b", re.IGNORECASE)
_LAST_EVENT = re.compile(
    r"\b(?:last|latest|most\s+recent|recent|current)\s+(?:\w+\s+)?(?:event|alarm|alert)\b",
    re.IGNORECASE)


def _focus(q: str, table) -> str | None:
    """The one item of `table` the question names; None for none or several ("BP and HR")."""
    hits = [name for name, rx in table if rx.search(q)]
    return hits[0] if len(hits) == 1 else None


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
# The patient's MEWS / early-warning score / criticality. STT hears "MEWS" as "NEWS" ("NEWS 5"),
# so "news" counts only before "score" or a number, never in "any news on this patient?". A
# definitional question about the score is a *document* question, not a lookup.
_MEWS = re.compile(
    r"\b(mews|news\s+(?:score|\d)|early warning|warning score|criticality)\b", re.IGNORECASE)
_DEFINITIONAL = re.compile(
    r"\b(how (?:is|are|do|does|to)|what does|means?|meaning|explain|definition|define|calculat\w*|"
    r"formula|criteria|protocol|guideline|escalat\w*|threshold|when should)\b",
    re.IGNORECASE,
)
# The worklist: pending (unacknowledged) High/Critical events. "work list" is how STT splits it.
_WORKLIST = re.compile(
    r"\b(work\s*-?\s*list|pending (?:alarms?|events?|alerts?)|un-?acknowledged|"
    r"outstanding (?:alarms?|events?|alerts?)|open (?:alarms?|alerts?))\b", re.IGNORECASE)
# Ranking / counting alarms across patients or beds.
_ALARM_COUNTS = re.compile(
    r"\b(?:most|highest(?: number of)?|how many|number of|count of|fewest|least)\b.*"
    r"\b(?:alarms?|alerts?|events?)\b|\b(?:alarms?|alerts?|events?)\b.*\b(?:per|by|for each)\s+"
    r"(?:bed|patient)\b", re.IGNORECASE)
_ALARM_WORD = re.compile(r"\b(alarms?|alerts?|events?)\b", re.IGNORECASE)
_MOST_SEVERE = re.compile(r"\b(?:most|highest)\s+(?:severe|critical|serious|urgent|priority)\b",
                          re.IGNORECASE)
# Event-history status words. `pending` is checked before `acknowledged` so "unacknowledged",
# "un-acknowledged" and "not acknowledged" are pending, never acknowledged.
_PENDING = re.compile(
    r"\b(work\s*-?\s*list|pending|un-?acknowledged|un-?acked|not (?:yet )?(?:been )?acknowledged|"
    r"outstanding (?:alarms?|events?|alerts?)|open (?:alarms?|alerts?|events?))\b", re.IGNORECASE)
_ACKED = re.compile(r"\b(?:acknowledged|acked)\b", re.IGNORECASE)
_WITHHELD = re.compile(r"\b(withheld|not alerted|suppressed)\b", re.IGNORECASE)
_HISTORY_WORD = re.compile(r"\b(history|episodes?|past|prior|previous)\b", re.IGNORECASE)
# A request to enumerate events ("list/show/say all…", "what are the…", "which…").
_LIST_VERB = re.compile(
    r"\b(list|show|say|tell|give|read|what (?:are|were)|which|any|all)\b", re.IGNORECASE)
_EVENTS_PLURAL = re.compile(r"\b(events|alarms|alerts|episodes|event history)\b", re.IGNORECASE)
_STRONG_LIST_VERB = re.compile(
    r"\b(list|show|say|tell me|give me|read(?: out)?|all|which|any)\b", re.IGNORECASE)
_CLINICAL_TOPIC = re.compile(
    r"\b(signs?|symptoms?|causes?|treat\w*|manag\w*|risks?|limits?|settings?|fatigue)\b",
    re.IGNORECASE)
# A target the clinician named that no rule could resolve ("events for Mr Smith", "events in the
# cardiac ward"). Listing everything instead would silently ignore it, so the agent declines.
_UNRESOLVED_TARGET = re.compile(r"\b(patient|bed|unit|room|ward|mr|mrs|ms)\b", re.IGNORECASE)
_NAMED_PATIENT = re.compile(r"\bPT\d{3,}\b", re.IGNORECASE)
_WHERE = re.compile(r"\b(which|what) bed\b|\bwhere\b|\bbed (?:is|does)\b", re.IGNORECASE)
# "this/current/same patient" — scopes a query to the session's patient (outbound call).
_THIS_PATIENT = re.compile(r"\b(this|current|same) patient(?:'s|s)?\b", re.IGNORECASE)


def history_status(q: str) -> str | None:
    """The event-history status a question asks for, or None (any status)."""
    if _WITHHELD.search(q):
        return "withheld"
    if _PENDING.search(q):
        return "pending"
    if _ACKED.search(q):
        return "acknowledged"
    return None


def looks_like_event_listing(query: str) -> bool:
    """A request to enumerate events/alarms. When no rule resolves one, the agent declines with a
    hint rather than letting a model guess which events were meant."""
    q = normalize_spoken_query(query or "")
    if _DEFINITIONAL.search(q) or _CLINICAL_TOPIC.search(q):
        return False  # "what are the alarm limits…", "signs of AFib episodes": document questions
    return bool(_EVENTS_PLURAL.search(q)) and bool(
        _STRONG_LIST_VERB.search(q) or history_status(q.lower()))


def asks_event_protocol(query: str, event_ref: str | None) -> bool:
    """'What is the protocol for this event?' / 'how do I treat this patient?' / 'what should I do
    now?': a document question about the CALLED event's rhythm (no rhythm, bed or patient named)."""
    if not event_ref:
        return False
    q = normalize_spoken_query(query or "")
    if event_type_from_text(q.lower()) or _BED_LABEL.search(q) or _NAMED_PATIENT.search(q):
        return False
    # Needs "this event / patient / rhythm": "how do you treat sepsis?" names its own topic.
    return bool(_EVENT_PROTOCOL.search(q) and _THIS_TARGET.search(q)
                and not _EMERGENCY_WORDS.search(q))


def is_patient_specific(query: str) -> bool:
    """About a particular patient or event ('this patient', 'his', a PT id, a bed number, 'the
    model'). With no graph query for it, such a question is never handed to free generation over
    the documents: live, the LLM invented patients, ages and IDs from archived event reports."""
    q = normalize_spoken_query(query or "")
    if _NAMED_PATIENT.search(q) or _BED_LABEL.search(q):
        return True  # a named patient or bed is always about that patient
    if event_type_from_text(q.lower()) or _DEFINITIONAL.search(q) or _NOT_EVENT_FOCUS.search(q):
        return False  # "how do I treat her AFib", "when should I escalate the patient": documents
    return bool(_PATIENT_TARGET.search(q))


def names_bed_label(query: str) -> bool:
    """The question names a bed by number ("bed 4", "bed four")."""
    return bool(_BED_LABEL.search(normalize_spoken_query(query or "")))


def names_unresolved_bed(query: str) -> bool:
    """The question names a bed but no bed number survived normalization ("on bed nein")."""
    q = normalize_spoken_query(query or "")
    # A bed as a place ("on bed nein", "bed number…"), not "a bed bath" / "the bed alarm".
    names_bed = re.search(r"\b(?:on|in|at|for)\s+(?:the\s+)?bed\b(?!\s+(?:bath|alarm|sensor|rail|"
                          r"side|pan|linen|exit))|\bbed\s+(?:number|no\.?)\b", q, re.IGNORECASE)
    return bool(names_bed) and not _BED_LABEL.search(q) and not _WHERE.search(q.lower())


def _history(*, now, status=None, patient=None, bed=None, etype=None, hours=None, critical=False):
    """(EVENT_HISTORY, params). `window_hours` only labels the answer; `since` filters."""
    params = {"status": status or "any"}
    if patient:
        params["patient_id"] = patient
    if bed:
        params["bed"] = bed
    if etype:
        params["event_type"] = etype
    if hours:
        params["since"] = now - hours * 3600
        params["window_hours"] = hours
    if critical:
        params["min_criticality"] = "High"
    return EVENT_HISTORY, params


def match_intent(query: str, *, now: float, patient_ref: str | None = None,
                 event_ref: str | None = None) -> tuple[str, dict] | None:
    """Map a natural-language query to (template_name, params), or None.

    Covers the operational matrix (T1–T10 + traversals). `patient_ref` scopes the session-relative
    vitals intent; `event_ref` (the worklist row the companion app has selected) makes "the event"
    mean *that* event rather than the patient's latest — without it, asking about a selected VF row
    answers from the patient's most recent event, which may be a different rhythm entirely.
    Bed/event-type-scoped intents read their target from the query text, so they work on the inbound
    text path too. Rules are ordered most- to least-specific (first match wins). The query is first
    normalized for spoken/STT phrasing (spelled acronyms, number-word bed labels) so voice and typed
    queries route identically.
    """
    query = normalize_spoken_query(query)
    q = query.lower()
    bed = (m.group(0) if (m := _BED_LABEL.search(query))
           else m.group(1) if (m := _BED.search(query)) else None)
    etype = event_type_from_text(q)
    # A patient the clinician names ("events for PT998224") scopes the patient intents below, over
    # the session's patient. It comes from the clinician's words, never from a model.
    named = m.group(0).upper() if (m := _NAMED_PATIENT.search(query)) else None
    if named:
        if _WHERE.search(q):
            return "patient_bed", {"patient_id": named}
        patient_ref = named
    elif patient_ref and _WHERE.search(q) and _THIS_TARGET.search(q) and "bed" in q:
        return "patient_bed", {"patient_id": patient_ref}  # "which bed is this patient on?"

    # T9 — ECG strips. Typed ("last AFib event"), else this patient's latest, else the global latest.
    if "ecg" in q or "strip" in q:
        if etype:
            return "ecg_strips_last_event_of_type", {"event_type": etype}
        if event_ref and not bed and not named:  # "the ECG strip for this event" on a call
            return "ecg_strip_for_event", {"event_uuid": event_ref}
        if patient_ref and _THIS_PATIENT.search(q):
            return "ecg_strips_for_patient_last_event", {"patient_id": patient_ref}
        return "ecg_strips_last_event", {}

    # T10 — HR/BP trend for the last event of a type ("HR and BP trend for the last VT event").
    if etype and ("trend" in q or ("hr" in q and "bp" in q)):
        return "trend_last_event_of_type", {"event_type": etype}

    # HR trend for the session patient's last event ("how was HR trending at this patient's event?").
    if (event_ref or patient_ref) and ("hr" in q or "heart rate" in q) \
            and ("trend" in q or "trending" in q) and not bed and not named:
        if event_ref:  # the called event's trend, not a newer event of the same patient
            return "hr_trend_for_event", {"event_uuid": event_ref}
        return "hr_trend_for_patient_last_event", {"patient_id": patient_ref}

    # List all patients who had an event of a type ("show all patients with an AFib event").
    if etype and _ALL_PATIENTS.search(q):
        return "patients_with_event_type", {"event_type": etype}

    # MEWS for the selected event, else the session patient's latest. Without this, "what is the
    # MEWS score?" fell through to document retrieval, where no patient data is in context and a
    # small model invented a patient (observed 2026-10-05).
    # A named bed wins: "MEWS score for the last event on bed seven" is about bed 7, not the call.
    bed_label = bed if bed and any(ch.isdigit() for ch in bed) else None

    # The called/selected event itself: rhythm, confidence, time, false positive, why alerted,
    # criticality, and what to do. Only with no other target named (a bed, a patient, a rhythm).
    # An emergency ("the patient is coding") goes to the documents' emergency steps, not to this.
    on_this_event = (event_ref and not bed and not named and not _DEFINITIONAL.search(q)
                     and not _EMERGENCY_WORDS.search(q) and not _NOT_EVENT_FOCUS.search(q))
    other_rules = bool(_VITALS.search(q) or _CLINICAL_HISTORY.search(q) or _EVENT_PROTOCOL.search(q))
    # First match wins here: "how confident is the model about this rhythm" is about confidence.
    if on_this_event and not etype and not other_rules and (focus := next(
            (name for name, rx in _EVENT_FOCUS if rx.search(q)), None)):
        return "event_summary", {"event_uuid": event_ref, "focus": focus}
    if on_this_event and not etype and _EVENT_ACTIONS.search(q) and not _EVENT_PROTOCOL.search(q) \
            and not _FOR_OTHER.search(q) \
            and not re.search(r"\b(?:outstanding|all|ward|every|across|other)\b", q):
        return "action_items_for_event", {"event_uuid": event_ref}
    if _MEWS.search(q) and not _DEFINITIONAL.search(q):
        if bed_label:
            return "bed_last_event", {"bed": bed_label, "focus": "mews"}
        if not bed and event_ref:
            return "mews_at_selected_event", {"event_uuid": event_ref}
        if not bed and patient_ref:
            return "mews_at_patient_last_event", {"patient_id": patient_ref}

    # The patient's clinical background (diagnoses, medications, surgeries, symptoms) — never the
    # monitor events ("prior events", "event history" stay listings below).
    history_patient = named or patient_ref
    history_focus = _focus(q, _HISTORY_FOCUS)
    # On a call the patient is implied: "any prior surgeries?" needs no "patient" (live, it fell to
    # the LLM, which invented "[P2] has a recorded CABG"). Not for cohort questions ("patterns
    # between age and event type") or a rhythm's symptoms ("symptoms of AFib").
    implied = (history_focus in ("medications", "surgeries", "allergies", "demographics")
               and not etype and not re.search(r"\b(pattern\w*|patients|cohort|all)\b", q))
    if (history_patient and _CLINICAL_HISTORY.search(q) and not _DEFINITIONAL.search(q)
            and not re.search(r"\b(events?|alarms?|alerts?|episodes?)\b", q)
            and (named or implied or _PATIENT_WORD.search(q)
                 or re.search(r"\b(medical|past|clinical)\s+history\b", q))):
        params = {"patient_id": history_patient}
        if history_focus:
            params["focus"] = history_focus
        return "patient_clinical_history", params


    # T5 — vitals at the event: a named bed wins (an explicit target beats the session's), then a
    # named rhythm for the session patient, then the selected/called event, then the patient's
    # latest.
    if _VITALS.search(q) and "trend" not in q:
        focus = {"focus": f} if (f := _focus(q, _VITAL_FOCUS)) else {}
        if bed:
            return "vitals_for_bed_last_event", {"bed": bed, **focus}
        if etype and patient_ref:
            return "vitals_at_patient_event_of_type", {
                "patient_id": patient_ref, "event_type": etype, "event_uuid": event_ref, **focus}
        if event_ref:
            return "vitals_at_selected_event", {"event_uuid": event_ref, **focus}
        if patient_ref:
            return "vitals_at_patient_last_event", {"patient_id": patient_ref, **focus}

    # (`bed_label` above: the generic _BED capture takes any word after "bed" — "which bed HAS
    # the…"; the rules below only trust a real label, as normalize_spoken_query writes it.)

    # Event history — listings answered from `list_events`, the same rows the app's Events tab and
    # `cli.events` show, with the one `pending` definition (alerted, not acknowledged/resolved).
    # Status ("unacknowledged", "acknowledged", "withheld") and scope (patient / bed / event type)
    # both come from the clinician's words.
    status = history_status(q)
    this_patient = patient_ref if _THIS_PATIENT.search(q) else None
    scope_patient = named or this_patient
    scoped = bool(scope_patient or bed_label or etype)
    listing = bool(_ALARM_WORD.search(q) or _HISTORY_WORD.search(q) or _WORKLIST.search(q))
    hours = int(m.group(1)) if (m := _HOURS.search(q)) else None
    critical = "critical" in q
    # "signs of AFib episodes", "how are alarms escalated": about the condition, not a listing.
    topic = bool(_DEFINITIONAL.search(q) or _CLINICAL_TOPIC.search(q))
    if topic:
        status = None
        listing = False
    if status and (listing or status == "pending") and not _MOST_SEVERE.search(q) \
            and not (_ALARM_COUNTS.search(q) and not scoped and not _WORKLIST.search(q)):
        return _history(now=now, status=status, patient=scope_patient, bed=bed_label, etype=etype,
                        hours=hours, critical=critical)
    # "events for this patient", "this patient's history / past / prior events" (any status).
    if this_patient and not topic and ("event" in q or _HISTORY_WORD.search(q)
                                       or _ALARM_WORD.search(q)):
        return _history(now=now, patient=this_patient, bed=bed_label, etype=etype, hours=hours,
                        critical=critical)
    # "all VT events", "how many AFib episodes on bed 2" — a typed listing.
    if etype and listing:
        return _history(now=now, patient=scope_patient, bed=bed_label, etype=etype, hours=hours,
                        critical=critical)

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
    if ("status" in q or "what happened" in q) and bed_label:
        return _history(now=now, bed=bed_label, hours=hours)

    # T7 — care protocol for a bed's last event. (Before T4 for the same reason.)
    if "protocol" in q and bed:
        return "protocol_for_bed_last_event", {"bed": bed}

    # T4 — event analysis report(s) for a bed (the noun "report"/"analysis").
    if _REPORT.search(q) and bed:
        return "reports_for_bed", {"bed": bed}

    # Worklist — "which bed has the most severe alarm?" (pending, most severe first). Unscoped
    # "unacknowledged alarms" is the event-history rule above. Hours narrow it.
    if _WORKLIST.search(q) or (_MOST_SEVERE.search(q) and not bed_label):
        hours = int(m.group(1)) if (m := _HOURS.search(q)) else None
        return "worklist", {"since": now - hours * 3600 if hours else 0}

    # "How many events today?" / "...in the last 6 hours?": a count, not the per-bed leaderboard.
    if re.search(r"\bhow\s+many\b", q) and _ALARM_WORD.search(q) \
            and not re.search(r"\b(per|by|each|which|most)\b", q) and not scoped:
        mins = int(m.group(1)) if (m := _MINUTES.search(q)) else None
        days = int(m.group(1)) if (m := re.search(r"\b(\d+)\s*days?\b", q)) else None
        if days is None and re.search(r"\b(?:this|past|last)\s+week\b", q):
            days = 7
        if mins and not hours:
            return "event_count_since", {"since": now - mins * 60, "window": f"in the last "
                                         f"{mins} minute{'s' if mins != 1 else ''}"}
        if hours:
            return "event_count_since", {"since": now - hours * 3600, "window": f"in the last "
                                         f"{hours} hour{'s' if hours != 1 else ''}"}
        if days:
            return "event_count_since", {"since": now - days * 86400, "window": f"in the last "
                                         f"{days} day{'s' if days != 1 else ''}"}
        if _TODAY.search(q):
            midnight = now - (now % 86400)  # UTC day, like every time the agent says
            return "event_count_since", {"since": midnight, "window": "today"}
        if not re.search(r"\b(?:week|month|yesterday|since)\b", q):  # a window we can't read
            return "event_count_since", {"since": 0, "window": "in total"}

    # Alarm counts — "which patient has the most alarms?", "how many alarms per bed?".
    if _ALARM_COUNTS.search(q) and not bed_label:
        return "alarm_counts", {}

    # "What was the last (reported) event on bed nine?": that one event, not the bed's listing.
    if bed_label and _LAST_EVENT.search(q):
        return "bed_last_event", {"bed": bed_label}

    # Alarms/events on a named bed ("what are the alarms in bed three?").
    if bed_label and _ALARM_WORD.search(q):
        return _history(now=now, bed=bed_label, hours=hours)

    # T6 — outstanding action items across patients.
    if "action item" in q or "outstanding" in q:
        return "outstanding_action_items", {}

    # Relationship lookup — co-morbidities of a named condition ("comorbidities of atrial fib").
    if "co-morbid" in q or "comorbid" in q:
        m = re.search(r"(?:of|with|for)\s+(.+?)\s*\??$", query, re.IGNORECASE)
        if m:
            return "comorbidity_neighborhood", {"condition_id": condition_id(m.group(1))}

    # Fallbacks for an explicit target the rules above did not cover: a named bed or patient is
    # answered from the graph, never by the LLM from conversation history (which, live, turned a
    # worklist summary's "9 Critical" into "bed 2 has … 9 Critical events" and dropped its VF).
    if bed_label:
        return _history(now=now, bed=bed_label, hours=hours)
    if named:
        return _history(now=now, patient=named, hours=hours)
    # An unscoped listing ("list all events", "show the alarms from the last 6 hours"), unless the
    # question names a target no rule resolved — then None, and the agent declines with a hint.
    if _EVENTS_PLURAL.search(q) and _STRONG_LIST_VERB.search(q) and not _UNRESOLVED_TARGET.search(q):
        return _history(now=now, hours=hours, critical=critical)

    return None


def _looks_like_cypher(query: str) -> bool:
    return bool(re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH|CALL|RETURN|UNWIND)\b", query))


def lookup(driver: GraphDriver, query: str, *, now: float) -> dict:
    """Resolve a query to rows: raw read-only Cypher as given, else an intent template."""
    if _looks_like_cypher(query):
        # Raw Cypher path — run only if read-only (refuses writes).
        return {"mode": "raw_cypher", "rows": driver.run_read_only(query)}

    intent = match_intent(query, now=now)
    if intent:
        name, params = intent
        return {"mode": "template", "template": name, "rows": run_template(driver, name, **params)}

    return {"mode": "declined", "rows": []}
