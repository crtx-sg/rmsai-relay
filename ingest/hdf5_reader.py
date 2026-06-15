"""HDF5 archive reader (Appendix A) → `SignalWindow`.

Maps the simulator's HDF5 layout onto the semantic `SignalWindow` contract. If the file format
changes, **this file changes — never `common/`**. The reader emits a raw `SignalWindow` with no
predicted `event_type` (that is the model's job, decision A).

Resilience (error matrix): a malformed/incomplete event is skipped with a logged error; the rest
of the file still processes. Window-math / missing-units violations reject *that event* loudly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py

from common.event_types import condition_code_to_name
from common.redacting_logger import get_redacting_logger
from common.schemas import GroundTruth, SignalWindow, Vital, VitalSample, WindowGeometry

from .rates import to_rational_rate

_log = get_redacting_logger("rmsai.ingest.hdf5")

# ECG lead order (decision D9 — all 7 leads). PPG/RESP are single-channel.
ECG_LEADS = ("ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX")
_DEFAULT_WAVEFORM_UNITS = "mV"
# One-sample tolerance when checking samples / rate == before + after seconds.
_WINDOW_MATCH_TOL_SAMPLES = 1


class ReaderError(Exception):
    """Raised for an event that cannot be safely read (rejected loudly)."""


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _load_extras(group: h5py.Group) -> dict:
    """Decode a group's `extras` UTF-8 JSON byte string; tolerate empty/missing."""
    if "extras" not in group:
        return {}
    raw = group["extras"][()]
    raw = _decode(raw)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _read_metadata(hf: h5py.File, *, strict_units: bool) -> dict:
    md = hf["metadata"]
    waveform_units = _decode(md["waveform_units"][()]) if "waveform_units" in md else None
    if waveform_units is None:
        # Decision C: waveform_units is an upstream addition. Fail loud only in strict mode;
        # default to mV otherwise (current simulator output omits it).
        if strict_units:
            raise ReaderError("metadata/waveform_units missing (strict mode)")
        waveform_units = _DEFAULT_WAVEFORM_UNITS
    return {
        "patient_id": _decode(md["patient_id"][()]),
        "rate_ecg": to_rational_rate(float(md["sampling_rate_ecg"][()])),
        "rate_ppg": to_rational_rate(float(md["sampling_rate_ppg"][()])),
        "rate_resp": to_rational_rate(float(md["sampling_rate_resp"][()])),
        "waveform_units": waveform_units,
        "before_s": float(md["seconds_before_event"][()]),
        "after_s": float(md["seconds_after_event"][()]),
        "alarm_offset_s": float(md["alarm_offset_seconds"][()]),
    }


def _check_window_math(group: str, n_samples: int, rate, before_s: float, after_s: float) -> None:
    # Compare in sample space with exact rationals (the simulator's RESP is 399 vs an ideal
    # 400, i.e. one sample short; a small sample-count tolerance absorbs that).
    expected_samples = (Fraction(before_s) + Fraction(after_s)) * Fraction(rate)
    if abs(n_samples - expected_samples) > _WINDOW_MATCH_TOL_SAMPLES:
        raise ReaderError(
            f"{group}: window math mismatch — {n_samples} samples at {float(rate):.4f} Hz, "
            f"expected ~{float(expected_samples):.2f} for {float(before_s + after_s):.2f}s"
        )


def read_event(hf: h5py.File, event_key: str, md: dict) -> SignalWindow:
    """Map one `event_<id>` group onto a `SignalWindow` (raises `ReaderError` on a bad event)."""
    grp = hf[event_key]

    if "uuid" not in grp or "timestamp" not in grp:
        raise ReaderError(f"{event_key}: missing uuid/timestamp")
    event_id = _decode(grp["uuid"][()])
    event_ts = float(grp["timestamp"][()])

    # Decision G: alarm_offset == seconds_before_event; assert and fail loud if violated.
    if abs(md["alarm_offset_s"] - md["before_s"]) > 1e-6:
        raise ReaderError(
            f"{event_key}: alarm_offset_seconds ({md['alarm_offset_s']}) != "
            f"seconds_before_event ({md['before_s']})"
        )
    start_ts = event_ts - md["before_s"]  # decision B

    signals: dict[str, list[float]] = {}
    sample_rates: dict[str, Any] = {}
    signal_quality: dict[str, float] = {}
    sample_counts: dict[str, int] = {}

    # --- ECG (7 leads) ---
    if "ecg" in grp:
        ecg = grp["ecg"]
        for lead in ECG_LEADS:
            if lead in ecg:
                arr = ecg[lead][:].astype(float).tolist()
                signals[lead] = arr
                sample_counts["ecg"] = len(arr)
        sample_rates["ecg"] = md["rate_ecg"]
        ecg_extras = _load_extras(ecg)
        if "signal_quality" in ecg_extras:
            signal_quality["ecg"] = float(ecg_extras["signal_quality"])
        pacer = None
        if "pacer_info" in ecg_extras or "pacer_offset" in ecg_extras:
            pacer = {
                "info": ecg_extras.get("pacer_info", 0),
                "offset": ecg_extras.get("pacer_offset", 0),
            }
    else:
        pacer = None

    # --- PPG / RESP (single channel each) ---
    for group_name, ds_name, rate_key in (("ppg", "PPG", "rate_ppg"), ("resp", "RESP", "rate_resp")):
        if group_name in grp and ds_name in grp[group_name]:
            arr = grp[group_name][ds_name][:].astype(float).tolist()
            signals[ds_name] = arr
            sample_counts[group_name] = len(arr)
            sample_rates[group_name] = md[rate_key]
            extras = _load_extras(grp[group_name])
            if "signal_quality" in extras:
                signal_quality[group_name] = float(extras["signal_quality"])

    # --- Window math self-consistency (per available signal group) ---
    for group_name, rate_key in (("ecg", "rate_ecg"), ("ppg", "rate_ppg"), ("resp", "rate_resp")):
        if group_name in sample_counts:
            _check_window_math(
                group_name, sample_counts[group_name], md[rate_key], md["before_s"], md["after_s"]
            )

    # --- Vitals + history ---
    vitals: dict[str, Vital] = {}
    vitals_history: dict[str, list[VitalSample]] = {}
    if "vitals" in grp:
        for vname in grp["vitals"]:
            vgrp = grp["vitals"][vname]
            if "value" in vgrp:
                vitals[vname] = Vital(
                    value=float(vgrp["value"][()]),
                    units=_decode(vgrp["units"][()]) if "units" in vgrp else "",
                    timestamp=float(vgrp["timestamp"][()]) if "timestamp" in vgrp else event_ts,
                )
            extras = _load_extras(vgrp)
            history = extras.get("history", [])
            if history:
                vitals_history[vname] = [
                    VitalSample(value=float(s["value"]), timestamp=float(s["timestamp"]))
                    for s in history
                    if "value" in s and "timestamp" in s
                ]

    # --- Ground truth (sim only) ---
    ground_truth = None
    cond = grp.attrs.get("condition")
    if cond is not None:
        cond = _decode(cond)
        ground_truth = GroundTruth(
            condition=condition_code_to_name(cond) or cond,
            heart_rate=float(grp.attrs["heart_rate"]) if "heart_rate" in grp.attrs else None,
            event_timestamp=(
                float(grp.attrs["event_timestamp"]) if "event_timestamp" in grp.attrs else None
            ),
        )

    return SignalWindow(
        patient_ref=md["patient_id"],
        event_id=event_id,
        start_timestamp=start_ts,
        event_timestamp=event_ts,
        signals=signals,
        sample_rates=sample_rates,
        waveform_units=md["waveform_units"],
        window=WindowGeometry(
            before_s=md["before_s"], after_s=md["after_s"], sample_counts=sample_counts
        ),
        vitals=vitals,
        vitals_history=vitals_history,
        signal_quality=signal_quality,
        pacer=pacer,
        ground_truth=ground_truth,
    )


def read_hdf5_file(path: str | Path, *, strict_units: bool = False) -> Iterator[SignalWindow]:
    """Yield one `SignalWindow` per readable event in the file (bad events are skipped + logged)."""
    path = Path(path)
    try:
        hf = h5py.File(path, "r")
    except OSError as exc:
        # Corrupt / truncated / unreadable file — log and yield nothing, never crash.
        _log.error("cannot open %s: %s", path.name, exc)
        return
    with hf:
        try:
            md = _read_metadata(hf, strict_units=strict_units)
        except (KeyError, ReaderError) as exc:
            _log.error("cannot read metadata in %s: %s", path.name, exc)
            return
        # Enumerate event_* groups; do NOT assume contiguous numbering.
        for event_key in sorted(k for k in hf.keys() if k.startswith("event_")):
            try:
                yield read_event(hf, event_key, md)
            except (ReaderError, KeyError, ValueError) as exc:
                _log.error("skipping %s/%s: %s", path.name, event_key, exc)
                continue
