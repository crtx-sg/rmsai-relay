"""Parameterized, tested, read-only operational query templates (Appendix B).

Operational queries run through these named templates (intent -> template). Free LLM
text-to-Cypher is a read-only allowlisted fallback only (see `lookup.py`). `$now`/`$since` are
computed by the caller. Every template is read-only.
"""

from __future__ import annotations

from .driver import GraphDriver, assert_read_only

TEMPLATES: dict[str, str] = {
    # T1 — Critical events in last N hours, by patient/bed/unit. "Critical" here means the
    # call-worthy events (criticality High or Critical, matching OUTBOUND_MIN_CRITICALITY's default),
    # i.e. the ones that warranted an alert — not strictly the 'Critical' tier, which would exclude
    # the High events clinicians are actually paged about.
    "critical_events_since": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.timestamp >= $since AND e.criticality IN ['High', 'Critical']
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event, e.criticality AS criticality, e.mews_risk AS mews,
               e.timestamp AS ts
        ORDER BY e.timestamp DESC
    """,
    # T2 — Positive (non-false-positive) events in last x minutes
    "positive_events_since": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.timestamp >= $since AND e.is_false_positive = false
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event,
               e.confidence AS confidence, e.timestamp AS ts
        ORDER BY e.timestamp DESC
    """,
    # T3 — Event status on a bed
    "event_status_on_bed": """
        MATCH (b:Bed {label: $bed})<-[:AT_BED]-(e:MonitoredEvent)
        RETURN e.timestamp AS ts, e.event_type AS reported_event,
               e.is_false_positive AS false_positive,
               e.ground_truth_condition AS actual_condition, e.status AS status
        ORDER BY e.timestamp DESC
    """,
    # T4 — Event analysis reports for a bed (refs; content from vector store)
    "reports_for_bed": """
        MATCH (b:Bed {label: $bed})<-[:AT_BED]-(e:MonitoredEvent)-[:HAS_REPORT]->(r:Report)
        RETURN e.timestamp AS ts, e.event_type AS event,
               r.id AS report_id, r.uri AS report_uri, r.summary AS summary
        ORDER BY e.timestamp DESC
    """,
    # T5 — Vitals at the time of a specific event
    "vitals_at_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        RETURN e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.timestamp AS ts
    """,
    # Vitals snapshot for a patient's most recent event — backs the "vitals at the time of the
    # event" chat intent, where "the event" is implicitly the latest one for the scoped patient.
    "vitals_at_patient_last_event": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event_type, e.criticality AS criticality, e.mews_risk AS mews_risk,
               e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.timestamp AS ts
    """,
    # Vitals snapshot for a bed's most recent event — the bed-scoped form of T5 ("vitals at the
    # event for the patient in Bed xx", where "the event" is that bed's latest MonitoredEvent).
    "vitals_for_bed_last_event": """
        MATCH (b:Bed {label: $bed})<-[:AT_BED]-(e:MonitoredEvent)
        WITH b, e ORDER BY e.timestamp DESC LIMIT 1
        MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event_type, e.criticality AS criticality, e.mews_risk AS mews_risk,
               e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.timestamp AS ts
    """,
    # "This patient" scoping (session patient) — critical (call-worthy) events for one patient.
    "critical_events_for_patient": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.criticality IN ['High', 'Critical']
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event, e.criticality AS criticality, e.mews_risk AS mews,
               e.timestamp AS ts
        ORDER BY e.timestamp DESC
    """,
    # "This patient" scoping — every event for one patient (any criticality), newest first.
    "events_for_patient": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event, e.criticality AS criticality,
               e.is_false_positive AS false_positive, e.timestamp AS ts
        ORDER BY e.timestamp DESC
    """,
    # T6 — Outstanding action items across all patients
    "outstanding_action_items": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)-[:HAS_ACTION]->(a:ActionItem)
        WHERE a.status = 'outstanding'
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, a.text AS action,
               a.priority AS priority, a.due_at AS due
        ORDER BY a.priority DESC
    """,
    # T7 — Care protocol for a bed's last event (hybrid: graph finds condition + protocol)
    "protocol_for_bed_last_event": """
        MATCH (b:Bed {label: $bed})<-[:AT_BED]-(e:MonitoredEvent)-[:OF_CONDITION]->(c:Condition)
        WITH e, c ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (cp:CareProtocol)-[:APPLIES_TO]->(c)
        OPTIONAL MATCH (cp)-[:HAS_STEP]->(s:ProtocolStep)
        RETURN c.name AS condition, e.event_type AS last_event,
               cp.title AS protocol, cp.source AS protocol_source,
               collect({order: s.order, kind: s.kind, text: s.text}) AS steps
    """,
    # T8 — Demographic / co-morbidity / symptom patterns vs event type
    "cohort_patterns": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.is_false_positive = false
        OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(c:Condition)
        OPTIONAL MATCH (p)-[:PRESENTS]->(s:Symptom)
        RETURN e.event_type AS event, p.gender AS gender,
               CASE WHEN p.age < 40 THEN '<40'
                    WHEN p.age < 65 THEN '40-64' ELSE '65+' END AS age_band,
               collect(DISTINCT c.name) AS comorbidities,
               collect(DISTINCT s.name) AS symptoms,
               count(DISTINCT e) AS n
        ORDER BY n DESC
    """,
    # T9 — ECG strips for a patient's last event of a given type (artifact refs)
    "ecg_strips_last_event": """
        MATCH (p:Patient {pseudonym: $patient})-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        WITH e ORDER BY e.timestamp DESC LIMIT 1
        RETURN e.uuid AS event, e.timestamp AS ts,
               e.signal_ref AS signal_ref, e.ecg_plot_ref AS ecg_plot
    """,
    # T10 — HR & BP trend for a patient's last event of a given type (plot ref)
    "trend_last_event": """
        MATCH (p:Patient {pseudonym: $patient})-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        WITH e ORDER BY e.timestamp DESC LIMIT 1
        RETURN e.uuid AS event, e.timestamp AS ts,
               e.vitals_plot_ref AS vitals_plot, e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp
    """,
    # ECG strips for the patient with the most recent event of a given type, across ALL patients
    # ("show ECG strips for the patient with the last reported AFib event"). The _of_type variants
    # find the patient by event type rather than requiring one up front.
    "ecg_strips_last_event_of_type": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.signal_ref AS signal_ref, e.ecg_plot_ref AS ecg_plot
    """,
    # HR & BP trend for the patient with the most recent event of a given type, across ALL patients.
    "trend_last_event_of_type": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.vitals_plot_ref AS vitals_plot, e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp
    """,
    # All patients who have had an event of a given type ("show all patients with an AFib event").
    "patients_with_event_type": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.timestamp AS ts, e.is_false_positive AS false_positive
        ORDER BY e.timestamp DESC
    """,
    # Relationship lookup — co-morbidity neighborhood of a condition
    "comorbidity_neighborhood": """
        MATCH (c:Condition {id: $condition_id})-[r:CO_MORBID_WITH]-(other:Condition)
        RETURN other.name AS comorbidity, r.confidence AS confidence,
               r.co_occurrence_count AS co_occurrence, r.source AS source
        ORDER BY r.confidence DESC
    """,
    # Relationship lookup — protocol/treatment guidance reachable from a condition
    "guidance_for_condition": """
        MATCH (c:Condition {id: $condition_id})
        OPTIONAL MATCH (cp:CareProtocol)-[:APPLIES_TO]->(c)
        OPTIONAL MATCH (g:Guideline)-[:APPLIES_TO]->(c)
        OPTIONAL MATCH (c)-[ind:INDICATES]->(t:Treatment)
        RETURN c.name AS condition,
               collect(DISTINCT cp.title) AS protocols,
               collect(DISTINCT g.title) AS guidelines,
               collect(DISTINCT {treatment: t.name, source: ind.source}) AS indicated_treatments
    """,
}

# All templates are verified read-only at import time.
for _name, _cypher in TEMPLATES.items():
    assert_read_only(_cypher)


def run_template(driver: GraphDriver, name: str, **params) -> list[dict]:
    """Run a named operational template with parameters (read-only)."""
    if name not in TEMPLATES:
        raise KeyError(f"unknown template: {name!r}")
    return driver.run_read(TEMPLATES[name], **params)
