"""JSON-safe serialization of a `DeviceEvent` for stdout + the event bus.

Raw signal arrays are excluded by default (they live in the HDF5 archive; the bus carries the
structured event + report). `Fraction` sample rates are rendered as `"num/den"` strings.

`dict_to_event` is the inverse used by the bus consumer: it reconstructs a `DeviceEvent` from a
published payload. The classification already happened in the producer, so the reconstruction is
faithful for everything the downstream flow reads (persist + report + outbound); the fields the bus
intentionally drops (raw signals, vitals history, signal quality, pacer, ECG-vital correlations)
come back empty.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from common.criticality import criticality
from common.schemas import (
    ClinicalAnalysis,
    DeviceEvent,
    GroundTruth,
    MEWS,
    SignalWindow,
    Vital,
    VitalTrend,
    WindowGeometry,
)


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


def dict_to_event(payload: dict[str, Any]) -> DeviceEvent:
    """Reconstruct a `DeviceEvent` from a bus payload produced by `event_to_dict`.

    Raises `KeyError`/`ValueError` on a malformed payload (the consumer treats that as a
    poison message). Bus-dropped fields (signals, vitals_history, signal_quality, pacer,
    correlations) are restored empty.
    """
    win = payload["window"]
    gt = payload.get("ground_truth")
    window = SignalWindow(
        patient_ref=payload["patient_ref"],
        event_id=payload["event_id"],
        start_timestamp=float(payload["start_timestamp"]),
        event_timestamp=float(payload["event_timestamp"]),
        sample_rates={g: Fraction(r) for g, r in payload.get("sample_rates", {}).items()},
        waveform_units=payload.get("waveform_units", "mV"),
        window=WindowGeometry(
            before_s=float(win["before_s"]),
            after_s=float(win["after_s"]),
            sample_counts=win.get("sample_counts", {}),
        ),
        vitals={
            n: Vital(value=float(v["value"]), units=v.get("units", ""),
                     timestamp=float(v.get("timestamp", payload["event_timestamp"])))
            for n, v in payload.get("vitals", {}).items()
        },
        ground_truth=(
            GroundTruth(condition=gt["condition"], heart_rate=gt.get("heart_rate")) if gt else None
        ),
    )
    mews = payload["mews"]
    analysis = ClinicalAnalysis(
        mews=MEWS(score=int(mews["score"]), risk=mews["risk"]),
        vital_trends={
            n: VitalTrend(direction=t["direction"], p=t.get("p"))
            for n, t in payload.get("vital_trends", {}).items()
        },
        care_guidance=list(payload.get("care_guidance", [])),
    )
    return DeviceEvent(
        window=window,
        event_type=payload["event_type"],
        confidence=float(payload["confidence"]),
        is_false_positive=bool(payload["is_false_positive"]),
        low_confidence=bool(payload.get("low_confidence", False)),
        uncertain=bool(payload.get("uncertain", False)),
        analysis=analysis,
        report_md=payload.get("report_md", ""),
    )


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
