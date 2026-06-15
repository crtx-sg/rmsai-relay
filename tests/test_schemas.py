"""Contract schemas: validation rules + import purity."""

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction

import pytest
from pydantic import ValidationError

from common.schemas import (
    ClinicalAnalysis,
    DeviceEvent,
    MEWS,
    RetrievalResult,
    SignalWindow,
    WindowGeometry,
)


def _window() -> SignalWindow:
    return SignalWindow(
        patient_ref="PT1",
        event_id="e1",
        start_timestamp=0.0,
        event_timestamp=6.0,
        signals={"ECG1": [0.0, 1.0]},
        sample_rates={"resp": Fraction(100, 3)},
        window=WindowGeometry(before_s=6.0, after_s=6.0),
    )


def test_signalwindow_carries_rational_rate():
    w = _window()
    assert w.sample_rates["resp"] == Fraction(100, 3)


def test_signalwindow_has_no_event_type_field():
    assert "event_type" not in SignalWindow.model_fields


def test_device_event_rejects_unknown_type():
    analysis = ClinicalAnalysis(mews=MEWS(score=1, risk="Low"))
    with pytest.raises((ValidationError, ValueError)):
        DeviceEvent(
            window=_window(),
            event_type="NOT_A_CLASS",
            confidence=0.9,
            is_false_positive=False,
            analysis=analysis,
        )


def test_device_event_accepts_known_type():
    analysis = ClinicalAnalysis(mews=MEWS(score=0, risk="Low"))
    ev = DeviceEvent(
        window=_window(),
        event_type="NORMAL_SINUS",
        confidence=0.95,
        is_false_positive=True,
        analysis=analysis,
    )
    assert ev.is_false_positive


def test_retrieval_result_vector_mode_has_no_relationships():
    r = RetrievalResult(query="q", mode="vector")
    assert r.relationships == []


def test_importing_common_does_not_pull_torch():
    # Run in a clean subprocess so other tests' imports can't pollute sys.modules.
    code = "import common, sys; assert 'torch' not in sys.modules, 'torch leaked into common'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
