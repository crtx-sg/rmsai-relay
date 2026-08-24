"""EventReport assembly — the Phase 1 event report grounded in graph patient context (D18).

Extends the per-event markdown report (from `inference/report.py`, rendered from the DeviceEvent's
typed fields) with a patient-context section pulled from the graph (history, co-morbidities,
symptoms, surgeries, age, gender). References the pseudonym only (G3).
"""

from __future__ import annotations

from common.config import DEFAULT, Config
from common.criticality import alert_basis, event_criticality
from common.schemas import DeviceEvent


def render_patient_context(ctx: dict) -> str:
    lines = ["## Patient context", ""]
    lines.append(f"- Demographics: {ctx.get('gender') or '?'}, age {ctx.get('age') or '?'}")
    lines.append(f"- Conditions: {', '.join(ctx.get('conditions') or []) or 'none recorded'}")
    lines.append(f"- Symptoms: {', '.join(ctx.get('symptoms') or []) or 'none recorded'}")
    lines.append(f"- Surgeries: {', '.join(ctx.get('surgeries') or []) or 'none recorded'}")
    lines.append(f"- Medications: {', '.join(ctx.get('medications') or []) or 'none recorded'}")
    return "\n".join(lines)


def build_event_report(event: DeviceEvent, patient_context: dict) -> str:
    """Combine the Phase 1 event report with the graph-derived patient context."""
    base = event.report_md.rstrip()
    return f"{base}\n\n{render_patient_context(patient_context)}\n"


def spoken_report(event: DeviceEvent, *, bed: str | None = None, config: Config = DEFAULT) -> str:
    """A concise spoken alert for the outbound call (the full markdown is for the chart/vector)."""
    w = event.window
    a = event.analysis
    crit = event_criticality(event, config)
    where = f" on bed {bed}" if bed else ""
    guidance = a.care_guidance[0] if a.care_guidance else ""
    guidance_txt = f" Recommended: {guidance}." if guidance else ""
    # A vitals-driven alert must not be spoken as a rhythm finding: the vitals are what we know, the
    # rhythm is either contradicted (a confident NORMAL_SINUS) or unconfirmed (below the confidence
    # gate). Lead accordingly and say which, so the clinician never hears an uncertain rhythm
    # asserted as fact.
    basis, why = alert_basis(event, config)
    detected = event.event_type.replace("_", " ")
    if basis == "vitals" and event.event_type == config.criticality_normal_event:
        lead, override_txt = f"Detected {detected}", (
            f" Note: the ECG rhythm is classified as normal sinus, so this alert is driven by "
            f"the patient's vitals ({why}), not the rhythm."
        )
    elif basis == "vitals":
        lead, override_txt = f"Possible {detected}", (
            f" Note: this alert is driven by the patient's vitals ({why}). The ECG suggests "
            f"{detected}, but at {event.confidence:.0%} confidence treat the rhythm as unconfirmed."
        )
    else:
        lead, override_txt = f"Detected {detected}", ""
    return (
        f"Alert for patient {w.patient_ref}{where}. {lead}, "
        f"{crit} criticality, MEWS {a.mews.score} ({a.mews.risk}), "
        f"confidence {event.confidence:.0%}.{guidance_txt}{override_txt}"
    )


def report_summary(event: DeviceEvent) -> str:
    """A one-line summary stored on the Report node and used as a citation snippet."""
    w = event.window
    fp = " (false positive)" if event.is_false_positive else ""
    return (
        f"{event.event_type}{fp} for {w.patient_ref}, "
        f"MEWS {event.analysis.mews.score} ({event.analysis.mews.risk}), "
        f"confidence {event.confidence:.2f}"
    )
