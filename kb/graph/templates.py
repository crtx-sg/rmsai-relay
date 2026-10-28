"""Parameterized, tested, read-only operational query templates (Appendix B).

Operational queries run through these named templates (intent -> template); the LLM router can
only pick one by name, never write Cypher. `$now`/`$since` are
computed by the caller. Every template is read-only.
"""

from __future__ import annotations

from .driver import GraphDriver, assert_read_only

#: Criticality tiers that make an unacknowledged event part of the worklist ("High and above").
WORKLIST_LEVELS = ["High", "Critical"]
#: Lifecycle statuses that take an event OFF the worklist (kb/graph/events.py: reported /
#: acknowledged / notify_failed / resolved). An ALERTED event with any other status is pending
#: (`kb.graph.events.is_pending`); an event that was never alerted is withheld, not pending.
CLOSED_STATUSES = ["acknowledged", "resolved"]

TEMPLATES: dict[str, str] = {
    # Worklist — every pending event (alerted, not acknowledged), most severe first, then newest:
    # the same rows as the app's worklist. (It used to keep only High/Critical, so "what is on my
    # worklist?" said 8 while "which bed has the most severe alarm?" said 5.) `$since` = 0 means
    # "no time window"; a question naming hours ("in the last 6 hours") narrows it.
    "worklist": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE NOT coalesce(e.status, 'reported') IN $closed
              AND coalesce(e.alert_gate, false) AND e.timestamp >= $since
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event, e.criticality AS criticality,
               coalesce(e.status, 'reported') AS status,
               coalesce(e.alert_gate, false) AS alert_gate, e.timestamp AS ts
        ORDER BY CASE e.criticality WHEN 'Critical' THEN 0 WHEN 'High' THEN 1
                                    WHEN 'Medium' THEN 2 ELSE 3 END, e.timestamp DESC
    """,
    # Every patient, sickest first — "list patients by criticality", "patients for bed rounds".
    # Patients with open (alerted, unacknowledged) alerts come first, by the most severe open alert;
    # then by the latest event's MEWS band; then newest. The MEWS *score* (in `why_json`) breaks ties
    # within a band in the answer renderer. One row per patient, with the current bed.
    "patients_by_criticality": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC
        WITH p, collect(e) AS evs
        WITH p, evs[0] AS latest,
             [x IN evs WHERE coalesce(x.alert_gate, false)
                             AND NOT coalesce(x.status, 'reported') IN $closed] AS open
        WITH p, latest, open,
             reduce(top = null, x IN open |
                    CASE WHEN top IS NULL OR
                              (CASE x.criticality WHEN 'Critical' THEN 3 WHEN 'High' THEN 2
                                    WHEN 'Medium' THEN 1 ELSE 0 END) >
                              (CASE top.criticality WHEN 'Critical' THEN 3 WHEN 'High' THEN 2
                                    WHEN 'Medium' THEN 1 ELSE 0 END)
                         THEN x ELSE top END) AS worst
        OPTIONAL MATCH (p)-[r:ASSIGNED_TO]->(b:Bed) WHERE coalesce(r.current, true)
        WITH p, latest, open, worst, head(collect(b.label)) AS bed
        RETURN p.pseudonym AS patient, bed, size(open) AS open_alerts,
               worst.criticality AS open_criticality, worst.event_type AS open_event,
               latest.event_type AS latest_event, latest.criticality AS latest_criticality,
               latest.mews_risk AS mews_risk, latest.why_json AS why_json, latest.timestamp AS ts
        ORDER BY CASE worst.criticality WHEN 'Critical' THEN 3 WHEN 'High' THEN 2
                      WHEN 'Medium' THEN 1 WHEN 'Low' THEN 0 ELSE -1 END DESC,
                 CASE latest.mews_risk WHEN 'Critical' THEN 3 WHEN 'High' THEN 2
                      WHEN 'Medium' THEN 1 ELSE 0 END DESC,
                 latest.timestamp DESC
    """,
    # Alarm counts per patient + bed, most first: "which patient has the most alarms?"
    "alarm_counts": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        WITH p, b, count(e) AS alarms,
             sum(CASE WHEN coalesce(e.alert_gate, false)
                           AND NOT coalesce(e.status, 'reported') IN $closed THEN 1 ELSE 0 END)
                 AS unacknowledged,
             sum(CASE WHEN e.criticality IN $levels THEN 1 ELSE 0 END) AS high_or_critical
        RETURN p.pseudonym AS patient, b.label AS bed, alarms, unacknowledged, high_or_critical
        ORDER BY alarms DESC, unacknowledged DESC
        LIMIT 10
    """,
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
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        RETURN p.pseudonym AS patient, b.label AS bed, e.timestamp AS ts,
               e.event_type AS reported_event,
               e.is_false_positive AS false_positive,
               e.ground_truth_condition AS actual_condition, e.status AS status,
               coalesce(e.alert_gate, false) AS alert_gate
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
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.why_json AS why_json,
               e.timestamp AS ts
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
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.why_json AS why_json,
               e.timestamp AS ts
    """,
    # Vitals at the patient's event of a named type ("vitals at the time of the VT"): the scoped
    # event when it is of that type, else the patient's latest of that type.
    "vitals_at_patient_event_of_type": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent {event_type: $event_type})
        WITH p, e ORDER BY coalesce(e.uuid = $event_uuid, false) DESC, e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event_type, e.criticality AS criticality, e.mews_risk AS mews_risk,
               e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.why_json AS why_json,
               e.timestamp AS ts
    """,
    # MEWS for the selected worklist event / the session patient's latest event. The score and the
    # vitals that contributed to it live in the event's stored explanation (`why_json`, written for
    # every event at persist time); the answer renderer reads them from there and never voices the
    # raw JSON. Routed by match_intent on MEWS / early-warning / criticality questions.
    "mews_at_selected_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event_type,
               e.criticality AS criticality, e.mews_risk AS mews_risk, e.why_json AS why_json,
               e.timestamp AS ts
    """,
    "mews_at_patient_last_event": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event_type,
               e.criticality AS criticality, e.mews_risk AS mews_risk, e.why_json AS why_json,
               e.timestamp AS ts
    """,
    # A bed's most recent event in one answer: rhythm, time, criticality, MEWS (score from the stored
    # explanation), status and vitals. "What was the last event on bed nine?" / "MEWS for bed seven".
    "bed_last_event": """
        MATCH (b:Bed {label: $bed})<-[:AT_BED]-(e:MonitoredEvent)
        WITH b, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event_type,
               e.criticality AS criticality, e.mews_risk AS mews_risk, e.why_json AS why_json,
               coalesce(e.status, 'reported') AS status,
               coalesce(e.alert_gate, false) AS alert_gate,
               e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.timestamp AS ts
    """,
    # A patient's clinical background from the graph: diagnoses, medications, surgeries, symptoms.
    "patient_clinical_history": """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(c:Condition)
        WITH p, collect(DISTINCT c.name) AS conditions
        OPTIONAL MATCH (p)-[:PRESCRIBED]->(t:Treatment)
        WITH p, conditions, collect(DISTINCT t.name) AS medications
        OPTIONAL MATCH (p)-[:HAD_SURGERY]->(s:Surgery)
        WITH p, conditions, medications, collect(DISTINCT s.name) AS surgeries
        OPTIONAL MATCH (p)-[:PRESENTS]->(y:Symptom)
        RETURN p.pseudonym AS patient, p.age AS age, p.gender AS gender, conditions,
               medications, surgeries, collect(DISTINCT y.name) AS symptoms
    """,
    # --- the called / selected event, by uuid (never "the patient's latest", which can be a newer
    # event: on the 2026-10-14 call the patient's VT at 07:17 answered questions about the VF call) --
    # Rhythm, confidence, time, false-positive flag, criticality, MEWS and why it was (not) alerted.
    "event_summary": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event_type,
               e.confidence AS confidence, e.is_false_positive AS false_positive,
               e.criticality AS criticality, e.mews_risk AS mews_risk, e.why_json AS why_json,
               e.why AS why, coalesce(e.status, 'reported') AS status,
               coalesce(e.alert_gate, false) AS alert_gate, e.timestamp AS ts
    """,
    "action_items_for_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})-[:HAS_ACTION]->(a:ActionItem)
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, a.text AS action, a.priority AS priority,
               a.status AS status
        ORDER BY CASE a.priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END
    """,
    "ecg_strip_for_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.signal_ref AS signal_ref, e.ecg_plot_ref AS ecg_plot
    """,
    "hr_trend_for_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.hr AS hr, e.hr_history AS hr_history, e.hr_history_ts AS hr_history_ts
    """,
    # Each vital's recorded history (oldest-first) at an event: "last 3 respiratory readings".
    "vital_readings_for_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event_type, e.timestamp AS ts,
               e.hr_history AS hr, e.sbp_history AS sbp, e.dbp_history AS dbp,
               e.spo2_history AS spo2, e.rr_history AS rr, e.temp_history AS temp
    """,
    # How many events since a time ("how many events today?"): total, alerted and still pending.
    "event_count_since": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.timestamp >= $since
        RETURN count(e) AS events,
               sum(CASE WHEN coalesce(e.alert_gate, false) THEN 1 ELSE 0 END) AS alerted,
               sum(CASE WHEN coalesce(e.alert_gate, false)
                             AND NOT coalesce(e.status, 'reported') IN $closed THEN 1 ELSE 0 END)
                   AS pending,
               count(DISTINCT p) AS patients
    """,
    # Vitals snapshot for the event the companion app has *selected* — the same shape as
    # `vitals_at_patient_last_event`, keyed by event uuid instead of "the patient's latest", so a
    # chat scoped to an older worklist row answers about that row's rhythm and not a newer one.
    "vitals_at_selected_event": """
        MATCH (e:MonitoredEvent {uuid: $event_uuid})
        OPTIONAL MATCH (p:Patient)-[:HAD_EVENT]->(e)
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit,
               e.event_type AS event_type, e.criticality AS criticality, e.mews_risk AS mews_risk,
               e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp,
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.why_json AS why_json,
               e.timestamp AS ts
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
               e.spo2 AS spo2, e.rr AS rr, e.temp AS temp, e.why_json AS why_json,
               e.timestamp AS ts
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
               e.is_false_positive AS false_positive,
               coalesce(e.status, 'reported') AS status,
               coalesce(e.alert_gate, false) AS alert_gate, e.timestamp AS ts
        ORDER BY e.timestamp DESC
    """,
    # Where a named patient is: "which bed is PT998224 on?"
    "patient_bed": """
        MATCH (p:Patient {id: $patient_id})-[r:ASSIGNED_TO]->(b:Bed)
        WHERE coalesce(r.current, true)
        OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit)
        RETURN p.pseudonym AS patient, b.label AS bed, u.name AS unit
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
               [x IN collect({order: s.order, kind: s.kind, text: s.text})
                WHERE x.text IS NOT NULL] AS steps
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
    # T10 — HR & BP trend for a patient's last event of a given type (plot ref)
    "trend_last_event": """
        MATCH (p:Patient {pseudonym: $patient})-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE e.event_type = $event_type
        WITH e ORDER BY e.timestamp DESC LIMIT 1
        RETURN e.uuid AS event, e.timestamp AS ts,
               e.vitals_plot_ref AS vitals_plot, e.hr AS hr, e.sbp AS sbp, e.dbp AS dbp
    """,
    # ECG strips for the most recent event across ALL patients (no event type named) —
    # "show ECG strips for the patient with the last reported event".
    "ecg_strips_last_event": """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.signal_ref AS signal_ref, e.ecg_plot_ref AS ecg_plot
    """,
    # ECG strips for the session patient's most recent event — "ECG strips for this patient's event".
    "ecg_strips_for_patient_last_event": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.signal_ref AS signal_ref, e.ecg_plot_ref AS ecg_plot
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
    # HR trend (the recorded HR history series) at the session patient's most recent event —
    # "how was HR trending at the time of this patient's event?". POC: HR only (RR/SpO2/BP later).
    "hr_trend_for_patient_last_event": """
        MATCH (p:Patient {id: $patient_id})-[:HAD_EVENT]->(e:MonitoredEvent)
        WITH p, e ORDER BY e.timestamp DESC LIMIT 1
        OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
        RETURN p.pseudonym AS patient, b.label AS bed, e.event_type AS event, e.timestamp AS ts,
               e.hr AS hr, e.hr_history AS hr_history, e.hr_history_ts AS hr_history_ts
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
    # Structured care-protocol steps for a rhythm ("what is the protocol for AFib?"), in order.
    "care_protocol_steps": """
        MATCH (cp:CareProtocol {event_type: $event_type})-[:HAS_STEP]->(s:ProtocolStep)
        WITH cp, s ORDER BY s.order
        RETURN cp.title AS protocol, cp.source AS source, collect(s.text) AS steps
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


# Parameters a template gets unless the caller supplies them (e.g. the LLM router picks
# "worklist" with no params: no time window, the standard tiers/statuses).
TEMPLATE_DEFAULTS: dict[str, dict] = {
    "worklist": {"since": 0, "levels": WORKLIST_LEVELS, "closed": CLOSED_STATUSES},
    "alarm_counts": {"levels": WORKLIST_LEVELS, "closed": CLOSED_STATUSES},
    "event_count_since": {"closed": CLOSED_STATUSES},
    "patients_by_criticality": {"closed": CLOSED_STATUSES},
}


#: Not a Cypher string: backed by `kb.graph.events.list_events`, the same listing the app's Events
#: tab and `cli.events` show, so a spoken event list can be checked against them row for row.
#: Params: patient_id, bed, event_type, status (any/pending/acknowledged/resolved/withheld), since.
EVENT_HISTORY = "event_history"


def event_history_rows(driver: GraphDriver, *, patient_id=None, bed=None, event_type=None,
                       status="any", since=None, min_criticality=None,
                       window_hours=None) -> list[dict]:
    """`list_events` rows in the compact shape the answer renderers read, newest first.
    (`window_hours` only labels the answer; `since` is the filter.)"""
    from .events import list_events  # noqa: PLC0415 - events imports this module

    rows = list_events(driver, patient=patient_id, bed=bed, event_type=event_type,
                       status=status, since=since or None, min_criticality=min_criticality)
    return [{
        "ts": r.get("timestamp"), "patient": r.get("patient"), "bed": r.get("bed"),
        "event": r.get("event_type"), "confidence": r.get("confidence"),
        "criticality": r.get("criticality"),
        "status": ("pending" if r["pending"] else "withheld" if not r.get("alert_gate")
                   else r.get("status") or "reported"),
        "event_id": r.get("id"),
    } for r in rows]


_STATUS_WORDS = {"any": "events", "pending": "pending (unacknowledged) events",
                 "acknowledged": "acknowledged events", "resolved": "resolved events",
                 "withheld": "withheld (not alerted) events"}


def describe_event_history(params: dict) -> str:
    """The filter an event-history answer used, as words: 'pending (unacknowledged) events on bed
    Unit1-Bed02 in the last 6 hours'. Every answer states it, so the clinician can apply the same
    filter on the Events tab (or `cli.events`) and see the same rows."""
    from .events import normalize_bed  # noqa: PLC0415

    text = _STATUS_WORDS.get(params.get("status") or "any", "events")
    if params.get("min_criticality"):
        text = f"High or Critical {text}"
    if params.get("event_type"):
        text += f" of type {str(params['event_type']).replace('_', ' ').lower()}"
    if params.get("patient_id"):
        text += f" for patient {params['patient_id']}"
    if params.get("bed"):
        text += f" on bed {normalize_bed(params['bed'])}"
    if params.get("window_hours"):
        h = params["window_hours"]
        text += f" in the last {h} hour{'s' if h != 1 else ''}"
    return text


def run_template(driver: GraphDriver, name: str, **params) -> list[dict]:
    """Run a named operational template with parameters (read-only)."""
    if name == EVENT_HISTORY:
        return event_history_rows(driver, **params)
    if name not in TEMPLATES:
        raise KeyError(f"unknown template: {name!r}")
    return driver.run_read(TEMPLATES[name], **{**TEMPLATE_DEFAULTS.get(name, {}), **params})
