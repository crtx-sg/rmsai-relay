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
    extra: dict | None = None,
) -> str:
    """MERGE a MonitoredEvent (by uuid) + its links. Returns the event id (== uuid).

    `extra` is a flat map of additional scalar/list properties (provenance, model identity, the
    alert decision and its explanation, the evaluation outcome) set with `SET e += $extra`, so new
    traceability fields don't widen this signature. None values are dropped.
    """
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
    if extra:
        props = {k: v for k, v in extra.items() if v is not None}
        if props:
            driver.run_write("MATCH (e:MonitoredEvent {id:$uuid}) SET e += $props",
                             uuid=uuid, props=props)

    if bed is not None:
        _, bed_label = bed
        driver.run_write(
            "MATCH (e:MonitoredEvent {id:$uuid}), (b:Bed {id:$bed}) MERGE (e)-[:AT_BED]->(b)",
            uuid=uuid, bed=bed_label,
        )

    # The clinical condition link is the *prediction*. Ground truth is evaluation-only: linking it
    # would let the answer key leak into clinical graph answers on labelled data.
    cond = link_condition
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


def get_event_patient(driver: GraphDriver, uuid: str) -> str | None:
    """Return the patient pseudonym owning a MonitoredEvent, or None if the event is unknown.

    Doubles as an existence check (an acknowledge for an unknown event is refused) and yields the
    pseudonym the acknowledge is audited against (never the event uuid as a subject).
    """
    rows = driver.run_read(
        "MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent {id:$uuid}) RETURN p.id AS pid",
        uuid=uuid,
    )
    return rows[0]["pid"] if rows else None


def get_event_report_summary(driver: GraphDriver, uuid: str) -> str | None:
    """Return the stored one-line `Report.summary` for an event, or None if absent.

    Backs the companion app's speak-on-select (and the SIP flow): the summary is the concise text
    `report_summary()` persisted on the Report node — spoken as-is, no model call. Returns None for
    an unknown event or one with no report yet, so the caller simply stays silent (hard rule #8).
    """
    rows = driver.run_read(
        "MATCH (e:MonitoredEvent {id:$uuid})-[:HAS_REPORT]->(r:Report) RETURN r.summary AS summary",
        uuid=uuid,
    )
    return (rows[0]["summary"] or None) if rows else None


def get_event_artifacts(driver: GraphDriver, uuid: str) -> dict | None:
    """Resolve an event's materialized artifact refs + owning pseudonym, or None if unknown.

    Backs the authenticated artifact endpoint: the ECG-strip path (`ecg_plot_ref`), the HR series
    (`hr_history`/`hr_history_ts`), and the report file (`Report.uri`). The pseudonym is what the
    view is audited against.
    """
    rows = driver.run_read(
        """
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent {id:$uuid})
        OPTIONAL MATCH (e)-[:HAS_REPORT]->(r:Report)
        RETURN p.id AS patient, e.ecg_plot_ref AS ecg_plot_ref,
               e.hr_history AS hr_history, e.hr_history_ts AS hr_history_ts, r.uri AS report_uri
        """,
        uuid=uuid,
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "patient": r["patient"],
        "ecg_plot_ref": r["ecg_plot_ref"],
        "hr_history": r["hr_history"],
        "hr_history_ts": r["hr_history_ts"],
        "report_uri": r["report_uri"],
    }


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


def set_event_delivery(driver: GraphDriver, uuid: str, **delivery) -> None:
    """Record what actually reached the clinician, after dispatch (None values are skipped).

    Keys used: `delivered_app` (worklist push succeeded), `delivered_call` (call outcome:
    answered / no_answer / invalid), `delivered_sms` (SMS fallback: sms_delivered / sms_failed).
    The gate decision (`alert_gate`) says whether an alert was *due*; these say whether it *landed*.
    """
    props = {k: v for k, v in delivery.items() if v is not None}
    if props:
        driver.run_write("MATCH (e:MonitoredEvent {id:$uuid}) SET e += $props", uuid=uuid, props=props)


#: Properties returned by `eval_events`: one row per event, everything the performance view needs.
EVAL_FIELDS = (
    "id", "timestamp", "processed_at", "event_type", "confidence", "ground_truth_condition", "criticality", "status",
    "model_id", "model_classes", "eval_outcome", "eval_unscorable_reason", "alert_gate",
    "alert_reason_code", "alert_basis", "why", "delivered_app", "delivered_call", "delivered_sms",
    "source_kind", "source_dataset", "source_record", "source_subject", "source_sample",
    "source_label", "source_label_method", "source_label_purity", "source_package", "source_split",
)


def eval_events(driver: GraphDriver, *, since: float | None = None, model_id: str | None = None,
                dataset: str | None = None, labelled_only: bool = True) -> list[dict]:
    """Events for the performance view/summary, most recently processed first, with the pseudonym.

    Filters are optional; `since` applies to when the relay *processed* the event (`processed_at`;
    falls back to the recording `timestamp` for events stored before it existed). `labelled_only`
    keeps events that have a ground truth.
    """
    fields = ", ".join(f"e.{f} AS {f}" for f in EVAL_FIELDS)
    rows = driver.run_read(
        f"""
        MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
        WHERE ($since IS NULL OR coalesce(e.processed_at, e.timestamp) >= $since)
          AND ($model IS NULL OR e.model_id = $model)
          AND ($dataset IS NULL OR e.source_dataset = $dataset)
          AND (NOT $labelled OR e.ground_truth_condition IS NOT NULL)
        RETURN p.id AS patient, {fields}
        ORDER BY coalesce(e.processed_at, e.timestamp) DESC
        """,
        since=since, model=model_id, dataset=dataset, labelled=labelled_only,
    )
    return [dict(r) for r in rows]


def get_event_info(driver: GraphDriver, uuid: str) -> dict | None:
    """One event's traceability record for the detail panel (None if unknown).

    `EVAL_FIELDS` plus the full explanation (`why_json`), the source device, and the patient pseudonym.
    """
    fields = ", ".join(f"e.{f} AS {f}" for f in (*EVAL_FIELDS, "why_json", "source_device",
                                                    "alert_reason", "mews_risk"))
    rows = driver.run_read(
        f"MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent {{id:$uuid}}) "
        f"RETURN p.id AS patient, {fields}",
        uuid=uuid,
    )
    return dict(rows[0]) if rows else None
