"""The model-performance view, built from graph rows (`kb.graph.events.eval_events`), shared by
`cli.model_perf` (text) and the gateway's `POST /metrics` (JSON), so both always agree.

One summary per model, because mixing two models' predictions describes neither. Pure: no I/O.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

from inference.metrics import EvalRecord, summarize
from orchestrator.event_log import source_label

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([mhd])$")


def parse_since(text: str | None, *, now: float | None = None) -> float | None:
    """`30m` / `24h` / `7d` → that long ago; an ISO date/time; or an epoch number. None passes."""
    if not text:
        return None
    now = time.time() if now is None else now
    m = _DURATION.match(text.strip().lower())
    if m:
        return now - float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        raise ValueError(f"since {text!r}: use 30m / 24h / 7d, an ISO date, or epoch seconds")


def group_by_model(rows: list[dict]) -> dict[str, dict]:
    """`{model_id: {"rows": [...], "classes": [...]}}`, models in first-seen order."""
    out: dict[str, dict] = {}
    for r in rows:
        m = out.setdefault(r.get("model_id") or "unknown model",
                           {"rows": [], "classes": r.get("model_classes")})
        m["rows"].append(r)
    return out


def records(rows: list[dict]) -> list[EvalRecord]:
    return [EvalRecord(r["event_type"], r.get("ground_truth_condition"), r.get("confidence"),
                       dispatched=r.get("alert_gate")) for r in rows]


def row_source(r: dict) -> str:
    """Source label for a graph row (flat `source_*` properties)."""
    if not r.get("source_kind"):
        return "source unknown"
    return source_label({"kind": r["source_kind"], "dataset": r.get("source_dataset"),
                         "record": r.get("source_record"), "source_sample": r.get("source_sample"),
                         "split": r.get("source_split"), "device": r.get("source_device")})


def build_perf_view(rows: list[dict]) -> dict:
    """`{"models": [{model_id, summary, sources, events}], "events": [...]}` for the dashboard."""
    models = []
    for model_id, m in group_by_model(rows).items():
        models.append({
            "model_id": model_id,
            "summary": summarize(records(m["rows"]), m["classes"]),
            "sources": sorted({r.get("source_dataset") or r.get("source_kind") or "unknown"
                               for r in m["rows"]}),
        })
    events = [{
        "event_id": r.get("id"), "patient": r.get("patient"),
        "processed_at": r.get("processed_at") or r.get("timestamp"),
        "predicted": r.get("event_type"), "confidence": r.get("confidence"),
        "truth": r.get("ground_truth_condition"), "outcome": r.get("eval_outcome"),
        "criticality": r.get("criticality"), "model_id": r.get("model_id"),
        "alert": bool(r.get("alert_gate")), "reason_code": r.get("alert_reason_code"),
        "delivered_app": r.get("delivered_app"), "delivered_call": r.get("delivered_call"),
        "why": r.get("why"), "source": row_source(r), "dataset": r.get("source_dataset"),
    } for r in rows]
    return {"models": models, "events": events}


_DEMO_SOURCES = {"simulator", "ecg_sigma"}


def _vitals_why(reason: str | None) -> str | None:
    """The stored gate reason "vitals_alert (MEWS 6 >= threshold 3)" -> "MEWS 6 >= threshold 3"
    (what a live row carries as alert_reason); other reasons ("ok") are not a vitals explanation."""
    m = re.match(r"^vitals_alert \((.*)\)$", reason or "")
    return m.group(1) if m else None


def build_events_view(rows: list[dict], *, low_confidence_below: float | None = None) -> list[dict]:
    """`kb.graph.events.list_events` rows as JSON for the app's Events tab AND the worklist backfill.

    Keys match a live `type:"event"` inbox message (live.inbox.build_event_message), so a backfilled
    pending row renders exactly like one pushed when it happened, plus the history-only fields
    (pending, alert, reason_code, delivered, processed_at, model_id, dataset).
    """
    out = []
    for r in rows:
        conf = r.get("confidence")
        demo = r.get("source_kind") in _DEMO_SOURCES
        out.append({
            "event_id": r.get("id"), "patient": r.get("patient"), "bed": r.get("bed"),
            "unit": r.get("unit"), "event_type": r.get("event_type"), "ts": r.get("timestamp"),
            "processed_at": r.get("processed_at"), "criticality": r.get("criticality"),
            "status": r.get("status") or "reported", "pending": bool(r.get("pending")),
            "confidence": conf,
            "low_confidence": bool(low_confidence_below is not None and conf is not None
                                   and conf < low_confidence_below),
            "alert": bool(r.get("alert_gate")), "reason_code": r.get("alert_reason_code"),
            "alert_basis": r.get("alert_basis") or "rhythm",
            "alert_reason": _vitals_why(r.get("alert_reason")),
            "why": r.get("why"), "false_positive": r.get("is_false_positive"),
            "delivered": {"app": r.get("delivered_app"), "call": r.get("delivered_call"),
                          "sms": r.get("delivered_sms")},
            "source": row_source(r) if demo else None, "source_label": row_source(r),
            "dataset": r.get("source_dataset"), "model_id": r.get("model_id"),
            "truth": r.get("ground_truth_condition"), "outcome": r.get("eval_outcome"),
            "links": {k: {} for k in r.get("artifact_kinds") or []},
        })
    return out
