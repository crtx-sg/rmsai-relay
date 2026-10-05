"""`VitalsAnalysis` wrapper over `ecgtranscnn.mews`.

Computes MEWS (`calculate_mews`), per-vital trends, and rule-based ECG-vital correlation notes
(`correlate_ecg_vitals`) from a `SignalWindow`, returning our `ClinicalAnalysis` contract.
Statistical/rule-based today; swappable for a learned model.

Trends: the vendored Mann-Kendall test supplies the p-value (and per-sample Sen slope); the
direction is decided by the hospital's clinical-significance policy (`common.vitals_trends`:
minimum change + normal range), not by statistical significance alone.

Resilience (error matrix): fewer than 3 history samples → `insufficient_data` trend; missing
vitals → MEWS degrades rather than throwing.
"""

from __future__ import annotations

from common.config import DEFAULT
from common.interfaces import VitalsAnalysis
from common.redacting_logger import get_redacting_logger
from common.schemas import ClinicalAnalysis, MEWS, MEWSComponentScore, SignalWindow, VitalTrend
from common.vitals_trends import TrendPolicy, classify_trend, load_trend_policy

_log = get_redacting_logger("rmsai.inference.vitals")

# Vitals MEWS needs; map our vital names straight through (HDF5 uses the same names).
_MEWS_REQUIRED = ("HR", "Systolic", "RespRate", "Temp", "SpO2")
_TREND_VITALS = ("HR", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp")


def _latest_vitals(window: SignalWindow) -> dict[str, float]:
    return {name: v.value for name, v in window.vitals.items()}


class MewsVitalsAnalysis(VitalsAnalysis):
    def __init__(self, policy: TrendPolicy | None = None):
        # Default: this deployment's hospital policy. Loaded eagerly so a broken policy file fails
        # at startup rather than on the first event.
        self.policy = policy or load_trend_policy(DEFAULT.hospital_id, DEFAULT.hospital_config_dir)

    def _trend(self, name: str, samples: list) -> VitalTrend:
        from ecg_transcovnet.mews import _classify_direction, mann_kendall  # noqa: PLC0415

        pts = sorted((s.timestamp, s.value) for s in samples)
        mk = mann_kendall([v for _, v in pts]) if len(pts) >= 3 else None
        rule = self.policy.vitals.get(name)
        if rule is None:  # a vital the policy doesn't cover: the vendored statistical verdict
            if mk is None:
                return VitalTrend(direction="insufficient_data")
            return VitalTrend(direction=_classify_direction(name, mk), p=mk.p_value, slope=mk.slope)
        v = classify_trend(pts, mk.p_value if mk else None, rule, self.policy.alpha)
        return VitalTrend(
            direction=v.direction, p=mk.p_value if mk else None, slope=mk.slope if mk else None,
            reason=v.reason, change=v.change, span_s=v.span_s, latest=v.latest,
            min_change=rule.min_change, normal_low=rule.normal_low, normal_high=rule.normal_high,
            unit=rule.unit,
        )

    def analyze(self, window: SignalWindow, event_type: str | None = None) -> ClinicalAnalysis:
        # Lazy import: ecgtranscnn.mews lives under a package whose __init__ pulls torch.
        from ecg_transcovnet.mews import calculate_mews, correlate_ecg_vitals  # noqa: PLC0415

        vitals = _latest_vitals(window)
        care_guidance: list[str] = []

        # --- MEWS (degrade gracefully if a component vital is missing) ---
        if all(k in vitals for k in _MEWS_REQUIRED):
            mews_res = calculate_mews(
                hr=vitals["HR"],
                systolic=vitals["Systolic"],
                resp_rate=vitals["RespRate"],
                temp_f=vitals["Temp"],
                spo2=vitals["SpO2"],
            )
            mews = MEWS(
                score=mews_res.total_score, risk=mews_res.risk_level,
                components=[MEWSComponentScore(name=c.name, value=c.value, score=c.score)
                            for c in mews_res.components],
            )
        else:
            missing = [k for k in _MEWS_REQUIRED if k not in vitals]
            _log.error("MEWS degraded — missing vitals %s", missing)
            mews = MEWS(score=0, risk="Low")
            mews_res = None
            care_guidance.append(f"Insufficient vitals for MEWS (missing {', '.join(missing)})")

        # --- Per-vital trends ---
        trends = {name: self._trend(name, window.vitals_history[name])
                  for name in _TREND_VITALS if window.vitals_history.get(name)}

        # --- ECG-vital correlation notes (needs the prediction) ---
        correlations: list[str] = []
        if mews_res is not None and event_type:
            correlations = correlate_ecg_vitals(event_type, vitals, mews_res)
        care_guidance.extend(correlations)

        return ClinicalAnalysis(
            mews=mews, vital_trends=trends, care_guidance=care_guidance, correlations=correlations
        )
