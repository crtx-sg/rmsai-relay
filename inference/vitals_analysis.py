"""`VitalsAnalysis` wrapper over `ecgtranscnn.mews`.

Computes MEWS (`calculate_mews`), per-vital Mann-Kendall trends (`assess_event_trends`), and
rule-based ECG-vital correlation notes (`correlate_ecg_vitals`) from a `SignalWindow`, returning
our `ClinicalAnalysis` contract. Statistical/rule-based today; swappable for a learned model.

Resilience (error matrix): too-few history samples → `insufficient_data` trend; missing vitals →
MEWS degrades rather than throwing.
"""

from __future__ import annotations

from common.interfaces import VitalsAnalysis
from common.redacting_logger import get_redacting_logger
from common.schemas import ClinicalAnalysis, MEWS, SignalWindow, VitalTrend

_log = get_redacting_logger("rmsai.inference.vitals")

# Vitals MEWS needs; map our vital names straight through (HDF5 uses the same names).
_MEWS_REQUIRED = ("HR", "Systolic", "RespRate", "Temp", "SpO2")
_TREND_VITALS = ("HR", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp")


def _latest_vitals(window: SignalWindow) -> dict[str, float]:
    return {name: v.value for name, v in window.vitals.items()}


def _history_dicts(window: SignalWindow) -> dict[str, list[dict]]:
    return {
        name: [{"value": s.value, "timestamp": s.timestamp} for s in samples]
        for name, samples in window.vitals_history.items()
    }


class MewsVitalsAnalysis(VitalsAnalysis):
    def analyze(self, window: SignalWindow, event_type: str | None = None) -> ClinicalAnalysis:
        # Lazy import: ecgtranscnn.mews lives under a package whose __init__ pulls torch.
        from ecg_transcovnet.mews import (  # noqa: PLC0415
            assess_event_trends,
            calculate_mews,
            correlate_ecg_vitals,
        )

        vitals = _latest_vitals(window)
        history = _history_dicts(window)
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
            mews = MEWS(score=mews_res.total_score, risk=mews_res.risk_level)
        else:
            missing = [k for k in _MEWS_REQUIRED if k not in vitals]
            _log.error("MEWS degraded — missing vitals %s", missing)
            mews = MEWS(score=0, risk="Low")
            mews_res = None
            care_guidance.append(f"Insufficient vitals for MEWS (missing {', '.join(missing)})")

        # --- Per-vital trends ---
        trends: dict[str, VitalTrend] = {}
        for t in assess_event_trends(history):
            trends[t.vital_name] = VitalTrend(direction=t.direction, p=t.p_value)
        # Vitals with history present but too short to assess -> insufficient_data.
        for name in _TREND_VITALS:
            if name not in trends and 0 < len(history.get(name, [])) < 2:
                trends[name] = VitalTrend(direction="insufficient_data")

        # --- ECG-vital correlation notes (needs the prediction) ---
        correlations: list[str] = []
        if mews_res is not None and event_type:
            correlations = correlate_ecg_vitals(event_type, vitals, mews_res)
        care_guidance.extend(correlations)

        return ClinicalAnalysis(
            mews=mews, vital_trends=trends, care_guidance=care_guidance, correlations=correlations
        )
