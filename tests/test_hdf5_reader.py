"""HDF5 reader → SignalWindow (Phase 1)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pytest

from common.schemas import SignalWindow
from ingest.hdf5_reader import ECG_LEADS, read_hdf5_file

_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
_FIXTURE = next(_FIXTURES.glob("*.h5"))
_FACTS = json.loads((_FIXTURES / "expected_facts.json").read_text())


@pytest.fixture(scope="module")
def windows() -> list[SignalWindow]:
    return list(read_hdf5_file(_FIXTURE))


def test_one_window_per_event(windows):
    assert len(windows) == len(_FACTS["events"])


def test_no_event_type_predicted(windows):
    # SignalWindow must not carry a predicted event_type (decision A).
    assert "event_type" not in windows[0].model_dump()


def test_fields_map(windows):
    w = windows[0]
    assert w.patient_ref == _FACTS["patient_id"]
    # 7 ECG leads + PPG + RESP present
    for lead in ECG_LEADS:
        assert lead in w.signals
    assert "PPG" in w.signals and "RESP" in w.signals
    assert w.waveform_units == "mV"  # defaulted (sim omits it)


def test_resp_rate_is_exact_rational(windows):
    assert windows[0].sample_rates["resp"] == Fraction(100, 3)
    assert windows[0].sample_rates["ecg"] == Fraction(200)


def test_window_math_holds(windows):
    w = windows[0]
    # ECG: 2400 / 200 = 12 s == before + after
    n_ecg = w.window.sample_counts["ecg"]
    assert abs(n_ecg / float(w.sample_rates["ecg"]) - (w.window.before_s + w.window.after_s)) < 1e-6


def test_start_timestamp_derived(windows):
    w = windows[0]
    assert abs(w.start_timestamp - (w.event_timestamp - w.window.before_s)) < 1e-6


def test_ground_truth_mapped_to_class_name(windows):
    # Fixture event_1001 ground truth code AFIB -> ATRIAL_FIBRILLATION
    conds = {w.ground_truth.condition for w in windows if w.ground_truth}
    assert "ATRIAL_FIBRILLATION" in conds
    assert "NORMAL_SINUS" in conds


def test_vitals_and_history(windows):
    w = windows[0]
    assert "HR" in w.vitals and w.vitals["HR"].units
    assert len(w.vitals_history["HR"]) >= 5


def test_strict_units_rejects_file_without_waveform_units():
    # Fixture omits metadata/waveform_units -> strict mode yields no events (logged).
    assert list(read_hdf5_file(_FIXTURE, strict_units=True)) == []


# --- Resilience: hand-built file with one good + one malformed event ---


def _build_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        md = f.create_group("metadata")
        md["patient_id"] = "PT0001"
        md["sampling_rate_ecg"] = 200.0
        md["sampling_rate_ppg"] = 75.0
        md["sampling_rate_resp"] = 33.33
        md["seconds_before_event"] = 6.0
        md["seconds_after_event"] = 6.0
        md["alarm_offset_seconds"] = 6.0
        # good event
        g = f.create_group("event_1001")
        g["uuid"] = "good-uuid"
        g["timestamp"] = 1000.0
        ecg = g.create_group("ecg")
        for lead in ECG_LEADS:
            ecg[lead] = np.zeros(2400, dtype=np.float32)
        ecg["extras"] = b""  # empty extras must not crash
        v = g.create_group("vitals").create_group("HR")
        v["value"] = 80.0
        v["units"] = "bpm"
        v["timestamp"] = 1000.0
        v["extras"] = json.dumps({"history": [{"value": 80, "timestamp": 1.0}]}).encode()
        g.attrs["condition"] = "AFIB"
        # malformed event: missing uuid + timestamp
        bad = f.create_group("event_1002")
        bad.create_group("ecg")


def test_skips_malformed_event_and_keeps_good(tmp_path):
    path = tmp_path / "PT0001_2026-01.h5"
    _build_file(path)
    windows = list(read_hdf5_file(path))
    assert len(windows) == 1
    assert windows[0].event_id == "good-uuid"
    assert windows[0].vitals["HR"].value == 80.0


def test_corrupt_file_yields_nothing_without_crashing(tmp_path):
    path = tmp_path / "corrupt.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"not a real hdf5 file" * 5)
    assert list(read_hdf5_file(path)) == []  # logged + empty, no exception
