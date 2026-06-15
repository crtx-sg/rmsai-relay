"""Per-event markdown report (D18) rendered from the `DeviceEvent`'s structured fields.

D18 rule: structure -> graph from typed fields; narrative -> vector. This renders the narrative.
The graph persistence (Phase 4) reads the typed fields directly, never by re-parsing this prose.
"""

from __future__ import annotations

from common.criticality import criticality
from common.schemas import DeviceEvent


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
            p = f" (p={t.p:.3f})" if t.p is not None else ""
            lines.append(f"  - {name}: {t.direction}{p}")
    if a.care_guidance:
        lines.append("- Care guidance:")
        lines += [f"  - {g}" for g in a.care_guidance]

    lines += ["", "## Vitals at event", ""]
    for name, v in sorted(w.vitals.items()):
        lines.append(f"- {name}: {v.value:g} {v.units}".rstrip())

    return "\n".join(lines) + "\n"
