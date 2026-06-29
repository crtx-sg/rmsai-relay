"""ECG strip rendering — synthetic samples in, PNG out (no HDF5, headless matplotlib)."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from common.config import DEFAULT
from common.schemas import SignalWindow, WindowGeometry
from inference.plotting import _pick_lead, render_ecg_strip


def _window(signals, event_id="evt-xyz"):
    return SignalWindow(
        patient_ref="PT9", event_id=event_id, start_timestamp=0.0, event_timestamp=1.0,
        signals=signals, sample_rates={"ecg": Fraction(250, 1)},
        window=WindowGeometry(before_s=2.0, after_s=2.0),
    )


def test_render_writes_png(tmp_path):
    cfg = replace(DEFAULT, plot_dir=str(tmp_path))
    sig = [math.sin(i / 5.0) for i in range(500)]
    path = render_ecg_strip(_window({"II": sig}), config=cfg)
    p = Path(path)
    assert p.is_file() and p.suffix == ".png" and p.stem == "evt-xyz"
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic — a real image


def test_render_none_without_signals(tmp_path):
    cfg = replace(DEFAULT, plot_dir=str(tmp_path))
    assert render_ecg_strip(_window({}), config=cfg) is None          # bus path (signals stripped)
    assert render_ecg_strip(_window({"II": []}), config=cfg) is None  # empty lead


def test_pick_lead_prefers_known_leads():
    assert _pick_lead({"V1": [1], "II": [1], "I": [1]}) == "II"   # II preferred
    assert _pick_lead({"foo": [1, 2]}) == "foo"                   # else first non-empty
    assert _pick_lead({"a": []}) is None                          # nothing usable
