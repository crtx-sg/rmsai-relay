"""ECGModel + VitalsAnalysis wrappers (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from inference.ecg_model import SIGNAL_LENGTH, get_ecg_model, window_to_lead_matrix
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import ECG_LEADS, read_hdf5_file

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


def _window():
    return next(read_hdf5_file(_FIXTURE))


# --- ECGModel ---


def test_lead_matrix_shape_and_order():
    m = window_to_lead_matrix(_window())
    assert m.shape == (len(ECG_LEADS), SIGNAL_LENGTH)
    assert m.dtype == np.float32


def test_lead_matrix_zero_fills_missing_leads():
    w = _window()
    del w.signals["aVR"]
    m = window_to_lead_matrix(w)
    idx = ECG_LEADS.index("aVR")
    assert not m[idx].any()  # zero-filled


def test_get_ecg_model_falls_back_to_stub_without_weights():
    model = get_ecg_model(checkpoint_path=None)
    assert isinstance(model, StubECGModel)
    model2 = get_ecg_model(checkpoint_path="does/not/exist.pt")
    assert isinstance(model2, StubECGModel)


def test_stub_predicts_valid_class_deterministically():
    w = _window()
    model = get_ecg_model()
    et, conf = model.predict(w)
    assert et in CLASS_NAMES
    assert model.predict(w) == (et, conf)


# --- VitalsAnalysis ---


def test_vitals_analysis_returns_mews_and_trends():
    w = _window()
    analysis = MewsVitalsAnalysis().analyze(w, event_type="ATRIAL_FIBRILLATION")
    assert analysis.mews.risk in {"Low", "Medium", "High", "Critical"}
    assert isinstance(analysis.mews.score, int)
    # Fixture has long histories -> at least one real trend assessed.
    assert analysis.vital_trends
    directions = {t.direction for t in analysis.vital_trends.values()}
    assert directions <= {"improving", "deteriorating", "stable", "insufficient_data"}


def test_vitals_analysis_degrades_without_required_vitals():
    w = _window()
    w.vitals.clear()
    analysis = MewsVitalsAnalysis().analyze(w, event_type="NORMAL_SINUS")
    assert analysis.mews.score == 0
    assert any("Insufficient vitals" in g for g in analysis.care_guidance)


def test_correlation_notes_use_prediction():
    w = _window()
    # Force an AFib-RVR style note: high HR + AFib prediction.
    w.vitals["HR"].value = 145.0
    analysis = MewsVitalsAnalysis().analyze(w, event_type="ATRIAL_FIBRILLATION")
    assert any("AFib" in note or "rate control" in note for note in analysis.correlations)
