"""Phase 1 pipeline: FP gate, thresholds, report, end-to-end DeviceEvent."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from common.config import Config
from common.ecg_model_stub import StubECGModel
from common.interfaces import ECGModel, VitalsAnalysis
from common.schemas import ClinicalAnalysis, MEWS, SignalWindow, Vital, WindowGeometry
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _FixedModel(ECGModel):
    def __init__(self, event_type: str, confidence: float):
        self._et, self._conf = event_type, confidence

    def predict(self, window):
        return self._et, self._conf


class _NullVitals(VitalsAnalysis):
    def analyze(self, window, event_type=None):
        return ClinicalAnalysis(mews=MEWS(score=0, risk="Low"))


def _bare_window() -> SignalWindow:
    return SignalWindow(
        patient_ref="PT1", event_id="e1", start_timestamp=0.0, event_timestamp=6.0,
        signals={"ECG1": [0.0]}, sample_rates={"ecg": Fraction(200)},
        window=WindowGeometry(before_s=6.0, after_s=6.0),
        vitals={"HR": Vital(value=80, units="bpm", timestamp=6.0)},
    )


# --- FP gate + thresholds ---


def test_normal_sinus_high_conf_is_false_positive():
    ev = process_window(_bare_window(), _FixedModel("NORMAL_SINUS", 0.95), _NullVitals())
    assert ev.is_false_positive and not ev.uncertain


def test_normal_sinus_low_conf_is_uncertain_not_fp():
    ev = process_window(_bare_window(), _FixedModel("NORMAL_SINUS", 0.70), _NullVitals())
    assert not ev.is_false_positive and ev.uncertain  # surfaced, not suppressed


def test_real_event_not_false_positive():
    ev = process_window(_bare_window(), _FixedModel("ATRIAL_FIBRILLATION", 0.9), _NullVitals())
    assert not ev.is_false_positive


def test_low_confidence_caveat_flag():
    ev = process_window(_bare_window(), _FixedModel("PVC", 0.50), _NullVitals())
    assert ev.low_confidence


def test_custom_thresholds_via_config():
    cfg = Config(fp_suppress_min_confidence=0.5)
    ev = process_window(_bare_window(), _FixedModel("NORMAL_SINUS", 0.6), _NullVitals(), cfg)
    assert ev.is_false_positive  # 0.6 >= 0.5 now suppresses


# --- End-to-end over the fixture ---


def test_end_to_end_emits_enriched_device_event():
    window = next(read_hdf5_file(_FIXTURE))
    ev = process_window(window, StubECGModel(), MewsVitalsAnalysis())
    assert ev.event_type
    assert ev.report_md.startswith("# Event Report")
    assert "MEWS" in ev.report_md
    # report carries the criticality + classification
    assert "Criticality" in ev.report_md
