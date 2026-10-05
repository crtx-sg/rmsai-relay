"""Clinical-significance trend policy (`common.vitals_trends`), hand-labelled cases.

Each case is a series a clinician would label by eye; the p-value comes from the vendored
Mann-Kendall test, exactly as the analyzer runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.vitals_trends import (
    AWAY_FROM_NORMAL,
    BELOW_MIN_CHANGE,
    NOT_SIGNIFICANT,
    TOWARD_NORMAL,
    WITHIN_NORMAL,
    VitalRule,
    classify_trend,
    load_trend_policy,
)

ALPHA = 0.05
HR = VitalRule(min_change=10, normal_low=51, normal_high=100, unit="bpm")
RR = VitalRule(min_change=4, normal_low=9, normal_high=14, unit="/min")
SPO2 = VitalRule(min_change=3, normal_low=94, normal_high=100, unit="%")
TEMP = VitalRule(min_change=1.0, normal_low=95.0, normal_high=101.1, unit="°F")


def _verdict(values, rule, step_s=300.0, times=None):
    from ecg_transcovnet.mews import mann_kendall

    ts = times or [i * step_s for i in range(len(values))]
    pts = list(zip(ts, [float(v) for v in values]))
    return classify_trend(pts, mann_kendall([v for _, v in pts]).p_value, rule, ALPHA)


def test_reported_rr_drift_is_significant_but_not_deterioration():
    # The case from the companion app: p≈0.0002, yet only ~1.6 breaths/min over the window.
    rr = [24, 23, 23, 23, 24, 23, 23, 24, 24, 24, 25, 24, 23, 25, 24, 25, 25, 24, 25, 25, 25, 24,
          25, 25, 25]
    v = _verdict(rr, RR)
    assert v.direction == "stable" and v.reason == BELOW_MIN_CHANGE
    assert 1.0 < v.change < 2.0 and v.span_s == 24 * 300


def test_rr_climbing_into_tachypnoea_is_deteriorating():
    v = _verdict([16, 17, 19, 20, 22, 23, 25, 26, 28, 30], RR)
    assert v.direction == "deteriorating" and v.reason == AWAY_FROM_NORMAL and v.change >= 10


def test_worsening_bradycardia_is_deteriorating():
    # The vendored rule calls any falling HR "improving".
    v = _verdict([52, 50, 48, 47, 45, 43, 41, 40, 38, 36], HR)
    assert v.direction == "deteriorating" and v.reason == AWAY_FROM_NORMAL


def test_bradycardia_recovering_into_normal_is_improving():
    # The vendored rule calls any rising HR "deteriorating".
    v = _verdict([38, 40, 43, 45, 48, 50, 53, 55, 58, 60], HR)
    assert v.direction == "improving" and v.reason == TOWARD_NORMAL


def test_large_rise_inside_normal_range_is_not_deterioration():
    v = _verdict([70, 72, 75, 77, 80, 82, 85, 87, 90, 92], HR)
    assert v.direction == "stable" and v.reason == WITHIN_NORMAL and v.change >= 20


def test_tachycardia_settling_but_still_high_is_improving():
    v = _verdict([140, 137, 135, 132, 130, 127, 125, 122, 120, 118], HR)
    assert v.direction == "improving" and v.reason == TOWARD_NORMAL


def test_desaturation_is_deteriorating():
    v = _verdict([98, 97, 97, 96, 95, 94, 93, 92, 91, 89], SPO2)
    assert v.direction == "deteriorating" and v.reason == AWAY_FROM_NORMAL and v.change < -3


def test_falling_into_hypothermia_is_deteriorating():
    v = _verdict([97.0, 96.6, 96.2, 95.9, 95.5, 95.1, 94.8, 94.4], TEMP)
    assert v.direction == "deteriorating" and v.reason == AWAY_FROM_NORMAL


def test_noise_without_trend_is_stable():
    v = _verdict([28, 22, 27, 23, 26, 22, 28, 23, 27, 22], RR)
    assert v.direction == "stable" and v.reason == NOT_SIGNIFICANT


def test_too_few_readings_is_insufficient():
    assert _verdict([20, 30], RR).direction == "insufficient_data"


def test_change_uses_time_not_sample_count():
    # Same values; the last two readings are an hour apart instead of 5 min. Sen's slope is per
    # second, so the change over the window reflects the real span.
    vals = [16, 18, 20, 22, 24]
    even = _verdict(vals, RR)
    uneven = _verdict(vals, RR, times=[0, 300, 600, 900, 4500])
    assert even.change == pytest.approx(8.0) and uneven.span_s == 4500
    assert uneven.change != even.change


# --- policy files (the vitals_trends section of config/hospitals/*.yaml) ---

_REPO_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "hospitals" / "default.yaml"


def _dir(tmp_path, site: str | None = None):
    (tmp_path / "default.yaml").write_text(_REPO_DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    if site is not None:
        (tmp_path / "h1.yaml").write_text(site, encoding="utf-8")
    return tmp_path


def test_repo_default_policy_loads_with_all_trend_vitals():
    p = load_trend_policy("")
    assert set(p.vitals) == {"HR", "RespRate", "SpO2", "Systolic", "Diastolic", "Temp"}
    assert p.vitals["RespRate"].min_change == 4 and p.source == "default.yaml"


def test_hospital_file_overrides_only_its_keys(tmp_path):
    d = _dir(tmp_path, "vitals_trends:\n  alpha: 0.01\n  vitals:\n    RespRate: {min_change: 3}\n")
    p = load_trend_policy("h1", d)
    assert p.alpha == 0.01 and p.source == "default.yaml + h1.yaml"
    assert p.vitals["RespRate"].min_change == 3 and p.vitals["RespRate"].normal_high == 14
    assert p.vitals["HR"].min_change == 10


def test_unknown_hospital_falls_back_to_default(tmp_path):
    assert load_trend_policy("h9", _dir(tmp_path)).source == "default.yaml"


@pytest.mark.parametrize("bad", ["../etc", "h1/x", "h 1"])
def test_hospital_id_cannot_escape_the_directory(tmp_path, bad):
    with pytest.raises(ValueError):
        load_trend_policy(bad, _dir(tmp_path))


@pytest.mark.parametrize("site", [
    "vitals_trends:\n  vitals:\n    HR: {min_change: 0}\n",        # non-positive threshold
    "vitals_trends:\n  vitals:\n    HR: {normal: [100, 51]}\n",    # inverted range
    "vitals_trends:\n  vitals:\n    Pulse: {min_change: 5}\n",     # new vital without a normal range
    "vitals_trends:\n  alpha: 1.5\n",
])
def test_invalid_policy_fails_loudly(tmp_path, site):
    with pytest.raises(ValueError):
        load_trend_policy("h1", _dir(tmp_path, site))


def test_analyzer_applies_the_policy_end_to_end():
    from common.schemas import VitalSample
    from inference.vitals_analysis import MewsVitalsAnalysis
    from common.vitals_trends import TrendPolicy

    rr = [24, 23, 23, 23, 24, 23, 23, 24, 24, 24, 25, 24, 23, 25, 24, 25, 25, 24, 25, 25]
    hist = {"RespRate": [VitalSample(value=v, timestamp=i * 300.0) for i, v in enumerate(rr)]}
    window = type("W", (), {"vitals": {}, "vitals_history": hist})()
    strict = MewsVitalsAnalysis(TrendPolicy(alpha=0.05, vitals={"RespRate": RR}, source="t"))
    loose = MewsVitalsAnalysis(TrendPolicy(
        alpha=0.05, vitals={"RespRate": VitalRule(1, 9, 14, "/min")}, source="t"))
    t = strict.analyze(window).vital_trends["RespRate"]
    assert t.direction == "stable" and t.reason == BELOW_MIN_CHANGE and t.p < 0.01
    assert t.min_change == 4 and (t.normal_low, t.normal_high) == (9, 14) and t.unit == "/min"
    # a hospital that treats a 1 /min change as significant gets the alert
    assert loose.analyze(window).vital_trends["RespRate"].direction == "deteriorating"
