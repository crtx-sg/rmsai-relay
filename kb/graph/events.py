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
            e.signal_ref=$signal_ref, e.ecg_plot_ref=$ecg_plot_ref, e.vitals_plot_ref=$vitals_plot_ref
        WITH e MATCH (p:Patient {id:$pid}) MERGE (p)-[:HAD_EVENT]->(e)
        """,
        uuid=uuid, ts=timestamp, etype=event_type, conf=confidence, fp=is_false_positive,
        gt=ground_truth_condition, mews=mews_risk, crit=crit, status=status,
        hr=vitals.get("hr"), sbp=vitals.get("sbp"), dbp=vitals.get("dbp"),
        spo2=vitals.get("spo2"), rr=vitals.get("rr"), temp=vitals.get("temp"),
        signal_ref=signal_ref, ecg_plot_ref=ecg_plot_ref, vitals_plot_ref=vitals_plot_ref,
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
