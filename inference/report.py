"""Per-event markdown report (D18) rendered from the `DeviceEvent`'s structured fields.

D18 rule: structure -> graph from typed fields; narrative -> vector. This renders the narrative.
The graph persistence (Phase 4) reads the typed fields directly, never by re-parsing this prose.
"""

from __future__ import annotations

from common.criticality import criticality
from common.schemas import DeviceEvent
from common.vitals_format import fmt_vital


def render_event_report(event: DeviceEvent) -> str:
    w = event.window
    a = event.analysis
    crit = criticality(event.event_type, a.mews.risk)

    lines: list[str] = [
        f"# Event Report — {w.patient_ref} / {w.event_id}",
        "",
        "## Classification",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Predicted | {event.event_type} |",
        f"| Confidence | {event.confidence:.2f} |",
        f"| False positive | {event.is_false_positive} |",
        f"| Criticality | {crit} |",
    ]
    if event.uncertain:
        lines.append("| Note | uncertain (NORMAL_SINUS below suppression threshold) |")
    if event.low_confidence:
        lines.append("| Note | low confidence — interpret with caution |")
    if w.ground_truth:
        lines.append(f"| Ground truth (sim) | {w.ground_truth.condition} |")

    lines += [
        "",
        "## Clinical Analysis",
        "",
        f"- MEWS: **{a.mews.score}** ({a.mews.risk})",
    ]
    if a.vital_trends:
        lines.append("- Vital trends:")
        for name, t in sorted(a.vital_trends.items()):
            lines.append(f"  - {name}: {t.direction}{_trend_detail(t)}")
    if a.care_guidance:
        lines.append("- Care guidance:")
        lines += [f"  - {g}" for g in a.care_guidance]

    lines += ["", "## Vitals at event", ""]
    for name, v in sorted(w.vitals.items()):
        lines.append(f"- {name}: {fmt_vital(name, v.value)} {v.units}".rstrip())

    return "\n".join(lines) + "\n"


_REASON_TEXT = {
    "not_significant": "no consistent trend",
    "below_min_change": "below the {min_change:g} {unit} threshold",
    "within_normal": "within normal {normal_low:g}–{normal_high:g}",
    "away_from_normal": "outside normal {normal_low:g}–{normal_high:g}, ≥ {min_change:g} {unit}",
    "toward_normal": "returning toward normal {normal_low:g}–{normal_high:g}",
}


def _trend_detail(t) -> str:
    """` — +1.6 /min over 52 min, below the 4 /min threshold (p=0.000)`; just ` (p=…)` for a trend
    without the clinical-policy fields."""
    p = "" if t.p is None else " (p<0.001)" if t.p < 0.001 else f" (p={t.p:.3f})"
    if t.change is None or t.reason not in _REASON_TEXT:
        return p
    why = _REASON_TEXT[t.reason].format(min_change=t.min_change or 0, unit=t.unit or "",
                                        normal_low=t.normal_low or 0, normal_high=t.normal_high or 0)
    span = f" over {round((t.span_s or 0) / 60)} min" if t.span_s else ""
    return f" — {round(t.change, 2):+g} {t.unit or ''}{span}, {why}{p}".replace("  ", " ")
