"""MQTT stream reader → `SignalWindow` (streaming parity, decision C).

A live bedside packet must carry an **equivalent metadata block** (units, rates, window
geometry) so a stream packet and an HDF5 event are interchangeable inputs to the same downstream
pipeline. This module defines that packet shape and the bidirectional mapping:

  * `window_to_packet(window)` — serialize (used by the simulated bedside publisher + tests).
  * `packet_to_window(packet)` — deserialize an incoming JSON packet to a `SignalWindow`.

The actual broker subscription is a thin wrapper over `packet_to_window` (Phase 1 broker runs
under the `telemetry` compose profile); `iter_packet_payloads` decodes raw JSON payloads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import Any

from common.schemas import GroundTruth, SignalWindow, Vital, VitalSample, WindowGeometry


def _rate_to_str(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def _str_to_rate(text: str) -> Fraction:
    return Fraction(text)


def window_to_packet(window: SignalWindow) -> dict[str, Any]:
    """Serialize a `SignalWindow` to a JSON-safe stream packet dict."""
    return {
        "patient_id": window.patient_ref,
        "event": {"uuid": window.event_id, "timestamp": window.event_timestamp},
        "metadata": {
            "waveform_units": window.waveform_units,
            "sample_rates": {g: _rate_to_str(r) for g, r in window.sample_rates.items()},
            "seconds_before_event": window.window.before_s,
            "seconds_after_event": window.window.after_s,
            "sample_counts": window.window.sample_counts,
        },
        "signals": window.signals,
        "signal_quality": window.signal_quality,
        "pacer": window.pacer,
        "vitals": {
            name: {"value": v.value, "units": v.units, "timestamp": v.timestamp}
            for name, v in window.vitals.items()
        },
        "vitals_history": {
            name: [{"value": s.value, "timestamp": s.timestamp} for s in samples]
            for name, samples in window.vitals_history.items()
        },
        "ground_truth": (
            {
                "condition": window.ground_truth.condition,
                "heart_rate": window.ground_truth.heart_rate,
                "event_timestamp": window.ground_truth.event_timestamp,
            }
            if window.ground_truth
            else None
        ),
    }


def packet_to_window(packet: dict[str, Any]) -> SignalWindow:
    """Map an incoming stream packet onto a `SignalWindow` (raises on missing required fields)."""
    md = packet["metadata"]
    event = packet["event"]
    before_s = float(md["seconds_before_event"])
    event_ts = float(event["timestamp"])

    gt = packet.get("ground_truth")
    ground_truth = (
        GroundTruth(
            condition=gt["condition"],
            heart_rate=gt.get("heart_rate"),
            event_timestamp=gt.get("event_timestamp"),
        )
        if gt
        else None
    )

    return SignalWindow(
        patient_ref=packet["patient_id"],
        event_id=event["uuid"],
        start_timestamp=event_ts - before_s,
        event_timestamp=event_ts,
        signals={k: [float(x) for x in v] for k, v in packet.get("signals", {}).items()},
        sample_rates={g: _str_to_rate(r) for g, r in md.get("sample_rates", {}).items()},
        waveform_units=md.get("waveform_units", "mV"),
        window=WindowGeometry(
            before_s=before_s,
            after_s=float(md["seconds_after_event"]),
            sample_counts=md.get("sample_counts", {}),
        ),
        vitals={
            name: Vital(value=float(v["value"]), units=v.get("units", ""),
                        timestamp=float(v.get("timestamp", event_ts)))
            for name, v in packet.get("vitals", {}).items()
        },
        vitals_history={
            name: [VitalSample(value=float(s["value"]), timestamp=float(s["timestamp"]))
                   for s in samples]
            for name, samples in packet.get("vitals_history", {}).items()
        },
        signal_quality={g: float(q) for g, q in packet.get("signal_quality", {}).items()},
        pacer=packet.get("pacer"),
        ground_truth=ground_truth,
    )


def iter_packet_payloads(payloads: Iterable[bytes | str]) -> Iterator[SignalWindow]:
    """Decode raw JSON payloads (e.g. MQTT message bodies) into `SignalWindow`s."""
    for payload in payloads:
        packet = json.loads(payload)
        yield packet_to_window(packet)
