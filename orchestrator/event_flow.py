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
from pathlib import Path

from common.config import DEFAULT, Config
from common.criticality import event_criticality
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


def write_report(report_md: str, event_id: str, *, config: Config = DEFAULT) -> str:
    """Materialize the report markdown to `{report_dir}/{event_id}.md`. Returns the file path (uri).

    The directory is gitignored (`data/` by default, like the audit log). Idempotent: a replayed
    event simply overwrites its own report. This is the durable source artifact behind the graph
    `Report` node; the vector index is the searchable copy.
    """
    path = Path(config.report_dir) / f"{event_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_md, encoding="utf-8")
    return str(path)


def traceability_props(event: DeviceEvent, config: Config = DEFAULT) -> dict:
    """Flat MonitoredEvent properties: data source, model, the alert decision + why, eval outcome.

    Everything the dashboard and logs need to answer "where did this come from, which model said
    so, why is it shown, and was it right?" without re-running anything.
    """
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    from inference.metrics import classify_outcome  # noqa: PLC0415
    from orchestrator.explain import explain_event  # noqa: PLC0415

    w = event.window
    p = w.provenance
    x = explain_event(event, config)
    gt = w.ground_truth.condition if w.ground_truth else None
    outcome = classify_outcome(event.event_type, gt, event.model_classes)
    return {
        "model_id": event.model_id, "model_classes": event.model_classes,
        "source_kind": p.kind if p else None, "source_device": p.device if p else None,
        "source_dataset": p.dataset if p else None, "source_record": p.record if p else None,
        "source_subject": p.subject if p else None,
        "source_sample": p.source_sample if p else None,
        "source_label": p.source_label if p else None,
        "source_label_method": p.label_method if p else None,
        "source_label_purity": p.label_purity if p else None,
        "source_package": p.package if p else None, "source_split": p.split if p else None,
        "alert_gate": x["decision"]["dispatch"], "alert_reason_code": x["decision"]["reason_code"],
        "alert_reason": x["decision"]["reason"], "alert_basis": x["basis"],
        "why": x["headline"], "why_json": json.dumps(x, ensure_ascii=False),
        # When the relay processed it. `timestamp` is the *recording* time, which for dataset
        # recordings can be years old, so "the last 24 h" filters on this instead.
        "processed_at": time.time(),
        "eval_outcome": outcome.code,
        "eval_unscorable_reason": outcome.reason or None,
    }


def process_device_event(
    event: DeviceEvent,
    driver: GraphDriver,
    vector: VectorRetriever,
    *,
    bed: tuple | None = None,
    generated_at: float = 0.0,
    config: Config = DEFAULT,
) -> EventFlowResult:
    """Persist + archive an inbound DeviceEvent. Returns a summary of what was written."""
    w = event.window
    crit = event_criticality(event, config)
    gt = w.ground_truth.condition if w.ground_truth else None
    actions = _action_items(event, crit)

    # ECG strip: use the producer-rendered path if present (bus path); else render now if the raw
    # samples are still in hand (direct path). HR history (oldest-first) backs the trend query.
    ecg_plot_ref = w.ecg_plot_ref
    if ecg_plot_ref is None and w.signals and config.ecg_plot_enabled:
        from inference.plotting import render_ecg_strip  # noqa: PLC0415

        ecg_plot_ref = render_ecg_strip(w, config=config)
    hr_samples = w.vitals_history.get("HR", [])
    hr_history = [s.value for s in hr_samples] or None
    hr_history_ts = [s.timestamp for s in hr_samples] or None

    # 1. persist MonitoredEvent (+ action items, dedupe by uuid)
    persist_monitored_event(
        driver, uuid=w.event_id, patient_id=w.patient_ref, timestamp=w.event_timestamp,
        event_type=event.event_type, confidence=event.confidence,
        is_false_positive=event.is_false_positive, mews_risk=event.analysis.mews.risk,
        ground_truth_condition=gt, status="reported", vitals=_vitals_snapshot(event), bed=bed,
        link_condition=event.event_type, action_items=actions,
        signal_ref=f"hdf5://{w.patient_ref}/{w.event_id}",
        ecg_plot_ref=ecg_plot_ref, hr_history=hr_history, hr_history_ts=hr_history_ts,
        extra=traceability_props(event, config),
    )

    # 2. patient context + 3. assemble report
    ctx = get_patient_context(driver, w.patient_ref)
    report_md = build_event_report(event, ctx)
    report_id = f"report:{w.event_id}"

    # 4. archive: materialize the report markdown to disk (durable, human-readable, survives a
    #    vector-store rebuild), record a Report node pointing at it, index the narrative for search,
    #    then mark indexed. The file is the source artifact; the vector index is the search copy.
    uri = write_report(report_md, w.event_id, config=config)
    persist_report(
        driver, event_uuid=w.event_id, report_id=report_id,
        uri=uri, summary=report_summary(event),
        generated_at=generated_at, index_status="pending",
    )
    vector.add_document(report_id, report_md)
    set_report_indexed(driver, report_id)

    return EventFlowResult(
        event_uuid=w.event_id, report_id=report_id, criticality=crit,
        report_md=report_md, action_items=len(actions),
    )
