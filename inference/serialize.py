"""JSON-safe serialization of a `DeviceEvent` for stdout + the event bus.

Raw signal arrays are excluded by default (they live in the HDF5 archive; the bus carries the
structured event + report). `Fraction` sample rates are rendered as `"num/den"` strings.
"""

from __future__ import annotations

from typing import Any

from common.criticality import criticality
from common.schemas import DeviceEvent


def event_to_dict(event: DeviceEvent, *, include_signals: bool = False) -> dict[str, Any]:
    w = event.window
    a = event.analysis
    payload: dict[str, Any] = {
        "patient_ref": w.patient_ref,
        "event_id": w.event_id,
        "event_timestamp": w.event_timestamp,
        "start_timestamp": w.start_timestamp,
        "event_type": event.event_type,
        "confidence": event.confidence,
        "is_false_positive": event.is_false_positive,
        "uncertain": event.uncertain,
        "low_confidence": event.low_confidence,
        "criticality": criticality(event.event_type, a.mews.risk),
        "mews": {"score": a.mews.score, "risk": a.mews.risk},
        "vital_trends": {n: {"direction": t.direction, "p": t.p} for n, t in a.vital_trends.items()},
        "care_guidance": a.care_guidance,
        "vitals": {
            n: {"value": v.value, "units": v.units, "timestamp": v.timestamp}
            for n, v in w.vitals.items()
        },
        "waveform_units": w.waveform_units,
        "sample_rates": {g: f"{r.numerator}/{r.denominator}" for g, r in w.sample_rates.items()},
        "window": {
            "before_s": w.window.before_s,
            "after_s": w.window.after_s,
            "sample_counts": w.window.sample_counts,
        },
        "ground_truth": (
            {"condition": w.ground_truth.condition, "heart_rate": w.ground_truth.heart_rate}
            if w.ground_truth
            else None
        ),
        "report_md": event.report_md,
    }
    if include_signals:
        payload["signals"] = w.signals
    return payload


def event_summary_line(event: DeviceEvent) -> dict[str, Any]:
    """A compact one-line summary (no report / vitals detail) for terminal scanning."""
    d = event_to_dict(event)
    return {
        "patient": d["patient_ref"],
        "event_id": d["event_id"],
        "event_type": d["event_type"],
        "confidence": round(d["confidence"], 3),
        "false_positive": d["is_false_positive"],
        "criticality": d["criticality"],
        "mews": d["mews"]["score"],
        "ground_truth": d["ground_truth"]["condition"] if d["ground_truth"] else None,
    }
