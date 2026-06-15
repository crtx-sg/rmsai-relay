"""Criticality lookup (gap G1).

`criticality(event_type, mews_risk) -> {Low, Medium, High, Critical}` — a small, reviewable
rule that gates the outbound call and protocol matching. Derived from the arrhythmia class and
the MEWS risk level; the more severe of the two wins.
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


def at_least(level: str, threshold: str) -> bool:
    """True iff `level` is at or above `threshold` on the criticality scale."""
    return _RANK.get(level, -1) >= _RANK.get(threshold, len(_ORDER))
