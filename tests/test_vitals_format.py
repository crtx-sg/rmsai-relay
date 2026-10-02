"""Display precision for vitals: integers, temperature to one decimal (stored values stay raw)."""

from __future__ import annotations

import pytest

from common.vitals_format import fmt_vital, round_vital
from orchestrator.event_log import event_log_line
from orchestrator.orchestrator import _answer_operational


@pytest.mark.parametrize("name,value,expect", [
    ("sbp", 110.115, "110"), ("dbp", 46.8304, "47"), ("spo2", 96.3793, "96"),
    ("Respiratory Rate", 14.999999850000002, "15"), ("HR", 98, "98"),
    ("temp", 99.2236, "99.2"), ("Temperature", 101.26, "101.3"), ("Temp", 98, "98.0"),
])
def test_fmt_vital(name, value, expect):
    assert fmt_vital(name, value) == expect


def test_non_numeric_passes_through():
    assert round_vital("hr", None) is None and round_vital("hr", "n/a") == "n/a"
    assert round_vital("hr", True) is True


def test_operational_answer_rounds_the_vitals():
    # the exact case reported: decimals leaked into the spoken/typed answer
    out = _answer_operational([{"patient": "PT992591", "bed": "Unit1-Bed03", "unit": "Unit1",
                                "event_type": "SINUS_TACHYCARDIA", "criticality": "Medium",
                                "mews_risk": "Low", "hr": 98.0, "sbp": 110.115, "dbp": 46.8304,
                                "spo2": 96.3793, "rr": 15.0, "temp": 99.2236,
                                "ts": 1737142440.0}])
    assert "The vitals at this event were hr 98; sbp 110; dbp 47; S P O 2 96; rr 15; temp 99.2" in out


def test_log_line_trend_samples_are_opt_in():
    trace = {"patient": "PT1", "source": "simulator", "predicted": "NORMAL_SINUS",
             "confidence": 0.9, "truth": None, "outcome": "UNSCORABLE", "criticality": "High",
             "gate": True, "reason_code": "fp_override", "why": "Shown because…",
             "trend_samples": {"HR": [88, 91, 95], "SpO2": []}}
    assert "trends:" not in event_log_line(trace)
    assert event_log_line(trace, samples=True).endswith("· trends: HR 88→91→95 (3 samples)")
