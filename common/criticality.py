"""Criticality lookup (gap G1).

`criticality(event_type, mews_risk) -> {Low, Medium, High, Critical}` — the intrinsic rule:
the more severe of the arrhythmia class and the MEWS risk level wins.

`assess_criticality(...)` / `event_criticality(event, config)` layer the **configurable escalation**
on top: criticality escalates to at least High when the event is not the normal baseline
(`CRITICALITY_NORMAL_EVENT`), the MEWS score is at/above `CRITICALITY_MEWS_THRESHOLD`, or a vital is
deteriorating (`CRITICALITY_ESCALATE_ON_DETERIORATING`). This is what gates the outbound call and
protocol matching. Escalation only raises to High — it never lowers an already-Critical event.
"""

from __future__ import annotations

from typing import Literal

Criticality = Literal["Low", "Medium", "High", "Critical"]

_ORDER: tuple[Criticality, ...] = ("Low", "Medium", "High", "Critical")
_RANK: dict[str, int] = {c: i for i, c in enumerate(_ORDER)}

# Intrinsic severity of each arrhythmia class, independent of vitals.
_EVENT_CRITICALITY: dict[str, Criticality] = {
    "NORMAL_SINUS": "Low",
    "SINUS_BRADYCARDIA": "Medium",
    "SINUS_TACHYCARDIA": "Medium",
    "ATRIAL_FIBRILLATION": "High",
    "ATRIAL_FLUTTER": "High",
    "PAC": "Low",
    "SVT": "High",
    "PVC": "Medium",
    "VENTRICULAR_TACHYCARDIA": "Critical",
    "VENTRICULAR_FIBRILLATION": "Critical",
    "LBBB": "Medium",
    "RBBB": "Low",
    "AV_BLOCK_1": "Low",
    "AV_BLOCK_2_TYPE1": "Medium",
    "AV_BLOCK_2_TYPE2": "High",
    "ST_ELEVATION": "Critical",
}

# MEWS risk maps straight onto the same scale.
_MEWS_CRITICALITY: dict[str, Criticality] = {
    "Low": "Low",
    "Medium": "Medium",
    "High": "High",
    "Critical": "Critical",
}


def criticality(event_type: str, mews_risk: str) -> Criticality:
    """Return the more severe of the event's intrinsic criticality and the MEWS risk."""
    event_c = _EVENT_CRITICALITY.get(event_type, "Medium")
    mews_c = _MEWS_CRITICALITY.get(mews_risk, "Low")
    return event_c if _RANK[event_c] >= _RANK[mews_c] else mews_c


def assess_criticality(
    event_type: str,
    mews_risk: str,
    mews_score: int,
    deteriorating: bool,
    *,
    normal_event: str = "NORMAL_SINUS",
    mews_threshold: int = 3,
    escalate_on_deteriorating: bool = True,
) -> Criticality:
    """Criticality under the configurable escalation rules.

    Starts from the intrinsic event/MEWS lookup (so VT/VF/ST stay Critical), then escalates to at
    least High when ANY of these hold: the event is not the `normal_event` baseline; the MEWS score
    is at/above `mews_threshold`; or a vital is `deteriorating` (when escalation on trend is on).
    Escalation only raises to High — it never lowers an already-Critical event.
    """
    base = criticality(event_type, mews_risk)
    escalate = (
        event_type != normal_event
        or mews_score >= mews_threshold
        or (escalate_on_deteriorating and deteriorating)
    )
    if escalate and _RANK[base] < _RANK["High"]:
        return "High"
    return base


def is_deteriorating(analysis) -> bool:
    """True iff any per-vital Mann-Kendall trend is deteriorating."""
    return any(t.direction == "deteriorating" for t in analysis.vital_trends.values())


def event_criticality(event, config) -> Criticality:
    """Configurable criticality for a `DeviceEvent` (duck-typed), reading thresholds from `config`."""
    return assess_criticality(
        event.event_type,
        event.analysis.mews.risk,
        event.analysis.mews.score,
        is_deteriorating(event.analysis),
        normal_event=config.criticality_normal_event,
        mews_threshold=config.criticality_mews_threshold,
        escalate_on_deteriorating=config.criticality_escalate_on_deteriorating,
    )


def at_least(level: str, threshold: str) -> bool:
    """True iff `level` is at or above `threshold` on the criticality scale."""
    return _RANK.get(level, -1) >= _RANK.get(threshold, len(_ORDER))
