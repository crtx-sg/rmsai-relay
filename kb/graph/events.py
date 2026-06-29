"""MonitoredEvent persistence — the queryable graph node behind a DeviceEvent.

Idempotent MERGE by `uuid` (dedupe across MQTT+HDF5 + replays). Links the event to patient, bed,
and condition, stores the inline vitals snapshot + criticality + lifecycle status, and attaches
`ActionItem`s from care guidance. Phase 4 extends this (report archival, FOLLOWED_BY chaining);
Phase 2B uses it to seed the operational-query test set.
"""

from __future__ import annotations

from common.criticality import criticality

from .driver import GraphDriver
from .entities import condition_id


def persist_monitored_event(
    driver: GraphDriver,
    *,
    uuid: str,
    patient_id: str,
    timestamp: float,
    event_type: str,
    confidence: float,
    is_false_positive: bool,
    mews_risk: str = "Low",
    ground_truth_condition: str | None = None,
    status: str = "reported",
    vitals: dict | None = None,
    bed: tuple | None = None,
    link_condition: str | None = None,
    action_items: list[dict] | None = None,
    signal_ref: str | None = None,
    ecg_plot_ref: str | None = None,
    vitals_plot_ref: str | None = None,
    hr_history: list | None = None,
    hr_history_ts: list | None = None,
) -> str:
    """MERGE a MonitoredEvent (by uuid) + its links. Returns the event id (== uuid)."""
    vitals = vitals or {}
    crit = criticality(event_type, mews_risk)
    driver.run_write(
        """
        MERGE (e:MonitoredEvent {id:$uuid})
        SET e.uuid=$uuid, e.timestamp=$ts, e.event_type=$etype, e.confidence=$conf,
            e.is_false_positive=$fp, e.ground_truth_condition=$gt, e.mews_risk=$mews,
            e.criticality=$crit, e.status=$status,
            e.hr=$hr, e.sbp=$sbp, e.dbp=$dbp, e.spo2=$spo2, e.rr=$rr, e.temp=$temp,
            e.signal_ref=$signal_ref, e.ecg_plot_ref=$ecg_plot_ref, e.vitals_plot_ref=$vitals_plot_ref,
            e.hr_history=$hr_history, e.hr_history_ts=$hr_history_ts
        WITH e MATCH (p:Patient {id:$pid}) MERGE (p)-[:HAD_EVENT]->(e)
        """,
        uuid=uuid, ts=timestamp, etype=event_type, conf=confidence, fp=is_false_positive,
        gt=ground_truth_condition, mews=mews_risk, crit=crit, status=status,
        hr=vitals.get("hr"), sbp=vitals.get("sbp"), dbp=vitals.get("dbp"),
        spo2=vitals.get("spo2"), rr=vitals.get("rr"), temp=vitals.get("temp"),
        signal_ref=signal_ref, ecg_plot_ref=ecg_plot_ref, vitals_plot_ref=vitals_plot_ref,
        hr_history=hr_history, hr_history_ts=hr_history_ts,
        pid=patient_id,
    )

    if bed is not None:
        _, bed_label = bed
        driver.run_write(
            "MATCH (e:MonitoredEvent {id:$uuid}), (b:Bed {id:$bed}) MERGE (e)-[:AT_BED]->(b)",
            uuid=uuid, bed=bed_label,
        )

    cond = link_condition or (ground_truth_condition if ground_truth_condition else None)
    if cond:
        driver.run_write(
            "MERGE (c:Condition {id:$cid}) SET c.name=coalesce(c.name,$name) "
            "WITH c MATCH (e:MonitoredEvent {id:$uuid}) MERGE (e)-[:OF_CONDITION]->(c)",
            cid=condition_id(cond.replace("_", " ")), name=cond, uuid=uuid,
        )

    for item in action_items or []:
        aid = f"{uuid}_{item['text'][:24]}"
        driver.run_write(
            "MATCH (e:MonitoredEvent {id:$uuid}) "
            "MERGE (a:ActionItem {id:$aid}) "
            "SET a.text=$text, a.priority=$priority, a.status=$status "
            "MERGE (e)-[:HAS_ACTION]->(a)",
            uuid=uuid, aid=aid, text=item["text"],
            priority=item.get("priority", "medium"), status=item.get("status", "outstanding"),
        )

    return uuid


def persist_report(
    driver: GraphDriver,
    *,
    event_uuid: str,
    report_id: str,
    uri: str,
    summary: str,
    generated_at: float,
    index_status: str = "pending",
) -> str:
    """MERGE a Report node and link it to its MonitoredEvent (idempotent). Returns the report id."""
    driver.run_write(
        "MATCH (e:MonitoredEvent {id:$euuid}) "
        "MERGE (r:Report {id:$rid}) "
        "SET r.uri=$uri, r.summary=$summary, r.generated_at=$gen, r.index_status=$idx "
        "MERGE (e)-[:HAS_REPORT]->(r)",
        euuid=event_uuid, rid=report_id, uri=uri, summary=summary,
        gen=generated_at, idx=index_status,
    )
    return report_id


def set_report_indexed(driver: GraphDriver, report_id: str) -> None:
    driver.run_write(
        "MATCH (r:Report {id:$rid}) SET r.index_status='indexed'", rid=report_id
    )


def set_event_status(driver: GraphDriver, uuid: str, status: str) -> None:
    """Update a MonitoredEvent's lifecycle status (reported/acknowledged/notify_failed/resolved)."""
    driver.run_write(
        "MATCH (e:MonitoredEvent {id:$uuid}) SET e.status=$status", uuid=uuid, status=status
    )


def get_patient_context(driver: GraphDriver, patient_id: str) -> dict:
    """Fetch a patient's demographics + history from the graph (for grounding event reports)."""
    rows = driver.run_read(
        """
        MATCH (p:Patient {id:$pid})
        OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(c:Condition)
        OPTIONAL MATCH (p)-[:PRESENTS]->(s:Symptom)
        OPTIONAL MATCH (p)-[:HAD_SURGERY]->(su:Surgery)
        OPTIONAL MATCH (p)-[:PRESCRIBED]->(t:Treatment)
        RETURN p.gender AS gender, p.age AS age,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT s.name) AS symptoms,
               collect(DISTINCT su.name) AS surgeries,
               collect(DISTINCT t.name) AS medications
        """,
        pid=patient_id,
    )
    if not rows:
        return {"gender": None, "age": None, "conditions": [], "symptoms": [],
                "surgeries": [], "medications": []}
    r = rows[0]
    return {
        "gender": r["gender"], "age": r["age"],
        "conditions": [c for c in r["conditions"] if c],
        "symptoms": [s for s in r["symptoms"] if s],
        "surgeries": [s for s in r["surgeries"] if s],
        "medications": [m for m in r["medications"] if m],
    }
