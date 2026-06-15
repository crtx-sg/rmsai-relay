"""Inbound event flow: DeviceEvent -> graph MonitoredEvent + archived EventReport.

Given a Phase 1 `DeviceEvent`, this:
  1. persists it as a `MonitoredEvent` (linked to patient/bed/condition, vitals snapshot,
     criticality, status) with `ActionItem`s derived from the care guidance;
  2. retrieves the patient context from the graph;
  3. assembles the `EventReport` (event report + patient context);
  4. archives it — a `Report` node linked to the event + the report narrative indexed into the
     vector store for later retrieval (graph is source of truth; vector index status tracked, G12).
"""

from __future__ import annotations

from dataclasses import dataclass

from common.criticality import criticality
from common.schemas import DeviceEvent
from kb.graph.driver import GraphDriver
from kb.graph.events import (
    get_patient_context,
    persist_monitored_event,
    persist_report,
    set_report_indexed,
)
from kb.vector.retriever import VectorRetriever

from .report import build_event_report, report_summary

# Map our vital names onto the MonitoredEvent snapshot fields.
_VITAL_FIELDS = {
    "HR": "hr", "Systolic": "sbp", "Diastolic": "dbp",
    "SpO2": "spo2", "RespRate": "rr", "Temp": "temp",
}


@dataclass
class EventFlowResult:
    event_uuid: str
    report_id: str
    criticality: str
    report_md: str
    action_items: int


def _vitals_snapshot(event: DeviceEvent) -> dict:
    return {field: event.window.vitals[name].value
            for name, field in _VITAL_FIELDS.items() if name in event.window.vitals}


def _action_items(event: DeviceEvent, crit: str) -> list[dict]:
    priority = "high" if crit in ("High", "Critical") else "medium"
    return [{"text": g, "priority": priority, "status": "outstanding"}
            for g in event.analysis.care_guidance]


def process_device_event(
    event: DeviceEvent,
    driver: GraphDriver,
    vector: VectorRetriever,
    *,
    bed: tuple | None = None,
    generated_at: float = 0.0,
) -> EventFlowResult:
    """Persist + archive an inbound DeviceEvent. Returns a summary of what was written."""
    w = event.window
    crit = criticality(event.event_type, event.analysis.mews.risk)
    gt = w.ground_truth.condition if w.ground_truth else None
    actions = _action_items(event, crit)

    # 1. persist MonitoredEvent (+ action items, dedupe by uuid)
    persist_monitored_event(
        driver, uuid=w.event_id, patient_id=w.patient_ref, timestamp=w.event_timestamp,
        event_type=event.event_type, confidence=event.confidence,
        is_false_positive=event.is_false_positive, mews_risk=event.analysis.mews.risk,
        ground_truth_condition=gt, status="reported", vitals=_vitals_snapshot(event), bed=bed,
        link_condition=gt or event.event_type, action_items=actions,
        signal_ref=f"hdf5://{w.patient_ref}/{w.event_id}",
    )

    # 2. patient context + 3. assemble report
    ctx = get_patient_context(driver, w.patient_ref)
    report_md = build_event_report(event, ctx)
    report_id = f"report:{w.event_id}"

    # 4. archive: Report node (pending) then index narrative into the vector store, then mark indexed
    persist_report(
        driver, event_uuid=w.event_id, report_id=report_id,
        uri=f"reports/{w.event_id}.md", summary=report_summary(event),
        generated_at=generated_at, index_status="pending",
    )
    vector.add_document(report_id, report_md)
    set_report_indexed(driver, report_id)

    return EventFlowResult(
        event_uuid=w.event_id, report_id=report_id, criticality=crit,
        report_md=report_md, action_items=len(actions),
    )
