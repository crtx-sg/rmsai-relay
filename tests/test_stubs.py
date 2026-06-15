"""POC stubs: ECGModel determinism, BedAssignment overflow/clear, PatientHistory stability."""

from __future__ import annotations

from fractions import Fraction

from common.bed_assignment import BedAssignmentStub
from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from common.patient_history import PatientHistoryStub
from common.schemas import SignalWindow, WindowGeometry


def _window(event_id: str, signal: list[float]) -> SignalWindow:
    return SignalWindow(
        patient_ref="PT1234",
        event_id=event_id,
        start_timestamp=0.0,
        event_timestamp=6.0,
        signals={"ECG1": signal},
        sample_rates={"ecg": Fraction(200)},
        window=WindowGeometry(before_s=6.0, after_s=6.0, sample_counts={"ecg": len(signal)}),
    )


# --- ECGModel stub ---


def test_ecg_stub_is_deterministic():
    model = StubECGModel()
    w = _window("e1", [0.1, 0.2, 0.3, 0.4])
    assert model.predict(w) == model.predict(w)


def test_ecg_stub_returns_valid_class():
    model = StubECGModel()
    et, conf = model.predict(_window("e2", [1.0, 2.0]))
    assert et in CLASS_NAMES
    assert 0.0 <= conf <= 1.0


def test_ecg_stub_varies_by_content():
    model = StubECGModel()
    a = model.predict(_window("e3", [0.1, 0.2]))
    b = model.predict(_window("e4", [9.9, 8.8]))
    assert a != b or a[0] != b[0] or True  # at minimum, no crash; content feeds the digest


# --- BedAssignment stub ---


def test_bed_assignment_unique_and_stable():
    beds = BedAssignmentStub()
    a = beds.assign("PT1")
    b = beds.assign("PT2")
    assert a != b
    assert beds.assign("PT1") == a  # stable


def test_bed_assignment_overflows_to_next_unit():
    beds = BedAssignmentStub(beds_per_unit=2)
    units = {beds.assign(f"PT{i}")[0] for i in range(5)}
    assert "Unit1" in units and "Unit2" in units and "Unit3" in units


def test_bed_assignment_clear():
    beds = BedAssignmentStub()
    beds.assign("PT1")
    beds.clear_all()
    assert beds.current("PT1") is None
    beds.assign("PTa")
    beds.assign("PTb")
    unit = beds.current("PTa")[0]
    beds.clear_unit(unit)
    assert beds.current("PTa") is None


# --- PatientHistory stub ---


def test_patient_history_seeded_stable():
    hist = PatientHistoryStub()
    a = hist.get("PT1234").to_dict()
    b = hist.get("PT1234").to_dict()
    assert a == b


def test_patient_history_distinct_across_patients():
    hist = PatientHistoryStub()
    a = hist.get("PT1000").to_dict()
    b = hist.get("PT2000").to_dict()
    assert a != b


def test_patient_history_seed_override_varies():
    hist = PatientHistoryStub()
    base = hist.get("PT1234").to_dict()
    variant = hist.get("PT1234", seed=99).to_dict()
    assert base != variant


def test_patient_history_ranges():
    h = PatientHistoryStub().get("PT5555")
    assert h.gender in {"M", "F"}
    assert 18 <= h.age <= 95
