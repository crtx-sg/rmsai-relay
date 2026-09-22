"""ECGModel + VitalsAnalysis wrappers (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from inference.ecg_model import (
    SIGNAL_LENGTH,
    EcgTransConvModel,
    get_ecg_model,
    window_to_lead_matrix,
)
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
    # A list where any member is missing is also a stub fallback, not a partial ensemble.
    model3 = get_ecg_model([_FOLDS[0], "does/not/exist.pt"])
    assert isinstance(model3, StubECGModel)


def test_stub_predicts_valid_class_deterministically():
    w = _window()
    model = get_ecg_model()
    et, conf = model.predict(w)
    assert et in CLASS_NAMES
    assert model.predict(w) == (et, conf)


# --- Real checkpoints (real_v2) -------------------------------------------------------------
# Weights are gitignored on both sides, so a tree without `make weights` skips rather than fails.

_REAL_V2 = Path(__file__).resolve().parents[1] / "external/ecgtranscnn/models/real_v2"
_FOLDS = [_REAL_V2 / f"fold{i}.pt" for i in range(5)]

_needs_weights = pytest.mark.skipif(
    not all(p.exists() for p in _FOLDS), reason="real_v2 weights absent — run `make weights`",
)


@_needs_weights
def test_real_ensemble_reads_its_contract_from_the_checkpoint():
    pytest.importorskip("ecg_transcovnet")
    model = get_ecg_model(_FOLDS)
    assert isinstance(model, EcgTransConvModel)
    # The head is the checkpoint's (13), and every name must be in our 16-name vocabulary.
    assert len(model.labels) == 13
    assert set(model.labels) <= set(CLASS_NAMES)
    # real_v2's head is exactly the first 13 of the vocabulary, in order — the property that lets
    # criticality/NL-matching/de-id keep working unchanged.
    assert model.labels == list(CLASS_NAMES[:13])


@_needs_weights
def test_real_ensemble_predicts_a_vocabulary_class():
    pytest.importorskip("ecg_transcovnet")
    model = get_ecg_model(_FOLDS)
    event_type, confidence = model.predict(_window())
    assert event_type in CLASS_NAMES
    assert 0.0 < confidence <= 1.0


@_needs_weights
def test_single_checkpoint_still_loads():
    """Back-compat with the one-path `--checkpoint model.pt` form."""
    pytest.importorskip("ecg_transcovnet")
    model = get_ecg_model(_FOLDS[0])
    assert isinstance(model, EcgTransConvModel)
    assert model.labels == list(CLASS_NAMES[:13])


@_needs_weights
def test_lead_mismatch_falls_back_to_stub(tmp_path, monkeypatch):
    """A checkpoint whose lead order differs must not silently feed the model transposed leads."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ecg_transcovnet")
    ckpt = torch.load(_FOLDS[0], weights_only=False, map_location="cpu")
    ckpt["leads"] = list(reversed(ckpt["leads"]))
    bad = tmp_path / "bad_leads.pt"
    torch.save(ckpt, bad)
    assert isinstance(get_ecg_model(bad), StubECGModel)


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
