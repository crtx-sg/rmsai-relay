"""EventReport assembly — the Phase 1 event report grounded in graph patient context (D18).

Extends the per-event markdown report (from `inference/report.py`, rendered from the DeviceEvent's
typed fields) with a patient-context section pulled from the graph (history, co-morbidities,
symptoms, surgeries, age, gender). References the pseudonym only (G3).
"""

from __future__ import annotations

import re

from common.config import DEFAULT, Config
from common.criticality import alert_basis, event_criticality
from common.schemas import DeviceEvent


def spoken_bed(label) -> str:
    """'Unit1-Bed05' or 'Unit1/Unit1-Bed05' -> 'bed 5' (one POC unit; the full label is in text)."""
    m = re.search(r"Bed0*(\d+)$", str(label or ""))
    return f"bed {m.group(1)}" if m else (f"bed {label}" if label else "an unknown bed")


def spoken_event(event) -> str:
    """'VENTRICULAR_TACHYCARDIA' -> 'ventricular tachycardia'; short codes (SVT, PVC) stay as-is."""
    text = str(event or "an event").replace("_", " ")
    return text if len(text) <= 4 else text.lower()


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


_SPOKEN_VITAL = {"Systolic": "systolic blood pressure", "Diastolic": "diastolic blood pressure",
                 "Temp": "temperature",
                 "SpO2": "oxygen saturation", "HR": "heart rate", "RespRate": "respiratory rate"}


def _spoken_mews(mews) -> str:
    """'MEWS 4 (Medium)', or 'MEWS unavailable, no systolic blood pressure or temperature'."""
    if mews.score is not None:
        return mews.label()
    names = [_SPOKEN_VITAL.get(m, m) for m in mews.missing]
    if not names:
        return "MEWS unavailable"
    return "MEWS unavailable, no " + (names[0] if len(names) == 1
                                       else ", ".join(names[:-1]) + " or " + names[-1])


def spoken_report(event: DeviceEvent, *, bed: str | None = None, config: Config = DEFAULT) -> str:
    """A concise spoken alert for the outbound call (the full markdown is for the chart/vector)."""
    if config.outbound_alert_language == "ja":
        return spoken_report_ja(event, bed=bed, config=config)
    w = event.window
    a = event.analysis
    crit = event_criticality(event, config)
    # Said, not read: "bed 2" and "ventricular fibrillation", not "bed Unit1/Unit1-Bed02" and
    # "VENTRICULAR FIBRILLATION" (the full label is on the worklist and in the chart).
    where = f" on {spoken_bed(bed)}" if bed else ""
    guidance = a.care_guidance[0] if a.care_guidance else ""
    guidance_txt = f" Recommended: {guidance}." if guidance else ""
    # A vitals-driven alert must not be spoken as a rhythm finding: the vitals are what we know, the
    # rhythm is either contradicted (a confident NORMAL_SINUS) or unconfirmed (below the confidence
    # gate). Lead accordingly and say which, so the clinician never hears an uncertain rhythm
    # asserted as fact.
    basis, why = alert_basis(event, config)
    detected = spoken_event(event.event_type)
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
        f"{crit} criticality, {_spoken_mews(a.mews)}, "
        f"confidence {event.confidence:.0%}.{guidance_txt}{override_txt}"
    )


# Japanese alert (outbound.alert_language: ja). Fixed templates over structured fields only, so no
# machine translation and no LLM; the caller still answers in English (STT stays English).
_JA_EVENT = {
    "NORMAL_SINUS": "正常洞調律", "SINUS_BRADYCARDIA": "洞性徐脈", "SINUS_TACHYCARDIA": "洞性頻脈",
    "ATRIAL_FIBRILLATION": "心房細動", "ATRIAL_FLUTTER": "心房粗動", "PAC": "心房期外収縮",
    "SVT": "上室性頻拍", "PVC": "心室期外収縮", "VENTRICULAR_TACHYCARDIA": "心室頻拍",
    "VENTRICULAR_FIBRILLATION": "心室細動", "LBBB": "左脚ブロック", "RBBB": "右脚ブロック",
    "AV_BLOCK_1": "1度房室ブロック", "AV_BLOCK_2_TYPE1": "2度房室ブロック、ウェンケバッハ型",
    "AV_BLOCK_2_TYPE2": "2度房室ブロック、モビッツ2型", "ST_ELEVATION": "ST上昇",
}
_JA_LEVEL = {"Critical": "重大", "High": "高", "Medium": "中", "Low": "低"}
JA_GREETING = "患者アラートのお知らせです。"
JA_PROMPT_CONFIRM = "アラートを聞くには、英語で「yes」と言ってください。"
# Call only (not the SMS fallback): the caller is heard in English.
JA_ACK_HINT = "確認する場合は、英語で「acknowledge」と言ってください。"


def _ja_why(why: str) -> str:
    """`common.criticality.vitals_override` reasons in Japanese (unknown ones pass through)."""
    m = re.fullmatch(r"MEWS (\d+) >= threshold (\d+)", why or "")
    if m:
        return f"MEWSスコア{m.group(1)}、基準値{m.group(2)}以上"
    return {"vitals deteriorating": "バイタルサインの悪化"}.get(why, why)


def spoken_report_ja(event: DeviceEvent, *, bed: str | None = None,
                     config: Config = DEFAULT) -> str:
    """`spoken_report` in Japanese: same facts and the same rhythm-vs-vitals wording rule. Care
    guidance (free English text) is left out; the clinician can ask for it in English."""
    w, a = event.window, event.analysis
    m = re.search(r"Bed0*(\d+)$", str(bed or ""))
    where = f"、{m.group(1)}番ベッド" if m else ""
    name = _JA_EVENT.get(event.event_type, spoken_event(event.event_type))
    crit = event_criticality(event, config)
    mews = (f"MEWSスコアは{a.mews.score}、リスク{_JA_LEVEL.get(a.mews.risk, a.mews.risk)}"
            if a.mews.score is not None else "MEWSは算出できません")
    pct = f"{event.confidence:.0%}".rstrip("%")
    basis, why = alert_basis(event, config)
    if basis == "vitals" and event.event_type == config.criticality_normal_event:
        lead, note = f"{name}を検出しました", (
            f"心電図のリズムは正常洞調律と判定されています。このアラートはリズムではなく、"
            f"患者のバイタルサイン（{_ja_why(why)}）によるものです。")
    elif basis == "vitals":
        lead, note = f"{name}の可能性があります", (
            f"このアラートは患者のバイタルサイン（{_ja_why(why)}）によるものです。心電図は{name}を"
            f"示唆していますが、信頼度が{pct}パーセントのため、リズムは未確定として扱ってください。")
    else:
        lead, note = f"{name}を検出しました", ""
    return (f"患者{w.patient_ref}{where}のアラートです。{lead}。"
            f"緊急度は{_JA_LEVEL.get(crit, crit)}、{mews}、信頼度は{pct}パーセントです。{note}")


def report_summary(event: DeviceEvent) -> str:
    """A one-line summary stored on the Report node and used as a citation snippet."""
    w = event.window
    fp = " (false positive)" if event.is_false_positive else ""
    return (
        f"{event.event_type}{fp} for {w.patient_ref}, "
        f"{event.analysis.mews.label()}, "
        f"confidence {event.confidence:.2f}"
    )
