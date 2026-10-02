"""Why the clinician is (or isn't) seeing an event: a structured explanation + one headline sentence.

Built only from the decision functions the relay actually runs: `should_call` (the gate),
`alert_basis` (rhythm vs vitals), `event_criticality`/`assess_criticality` (the level) and the
vitals analysis (MEWS sub-scores, Mann-Kendall trends). So the explanation can't drift from the
behaviour; it re-states the same rules, it doesn't re-decide them.

Unlike `vitals_override`, which stops at the first trigger, this lists **every** reason: MEWS at or
above the threshold *and* each deteriorating vital with its direction (rising/falling, from the trend
slope) and p-value. The headline leads with what the alert actually rests on: the rhythm when it
stands on its own, the vitals when it doesn't, so it never asserts a rhythm the classifier couldn't
stand behind.

The gate is evaluated with outbound forced on, as `cli.consume`/`cli.outbound` run it.
"""

from __future__ import annotations

from dataclasses import replace

from common.config import DEFAULT, Config
from common.criticality import alert_basis, criticality, event_criticality, is_deteriorating
from common.vitals_format import round_vital

_VITAL_NAMES = {"HR": "HR", "SpO2": "SpO₂", "Systolic": "systolic BP", "Diastolic": "diastolic BP",
                "RespRate": "resp rate", "Temp": "temp"}

#: Human phrasing for each gate reason code (the code is the prefix of `should_call`'s reason).
REASON_TEXT = {
    "ok": "rhythm finding",
    "vitals_alert": "vitals-driven (rhythm unconfirmed)",
    "fp_override": "vitals-driven (rhythm normal)",
    "false_positive": "normal rhythm, vitals calm",
    "low_confidence_arrhythmia": "rhythm unconfirmed, vitals calm",
    "below_threshold": "below the criticality threshold",
    "outbound_disabled": "alerts disabled",
}


_ACRONYMS = {"RBBB", "LBBB", "PVC", "PAC", "SVT", "AV", "ST"}
_MAX_TRENDS = 3  # in a headline; the structured explanation keeps them all


def _rhythm_name(event_type: str) -> str:
    """`VENTRICULAR_TACHYCARDIA` → `ventricular tachycardia`; acronyms stay upper (`RBBB`, `AV block 1`)."""
    return " ".join(w if w in _ACRONYMS else w.lower() for w in event_type.split("_"))


def _p(p: float | None) -> str:
    if p is None:
        return ""
    return " (p<0.001)" if p < 0.001 else f" (p={p:.2g})"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def explain_event(event, config: Config = DEFAULT) -> dict:
    """Structured explanation for one `DeviceEvent` (duck-typed); see the module docstring."""
    from orchestrator.outbound_flow import should_call  # noqa: PLC0415 (outbound_flow imports deps)

    gate_cfg = replace(config, outbound_enabled=True)
    dispatch, reason = should_call(event, gate_cfg)
    code = reason.split(" ", 1)[0]
    basis, _ = alert_basis(event, gate_cfg)
    a = event.analysis
    normal = event.event_type == config.criticality_normal_event

    # --- rhythm ---
    thr = config.outbound_min_arrhythmia_confidence
    if normal:
        status = "normal"
    elif event.confidence >= thr:
        status = "asserted"
    else:
        status = "unconfirmed"
    rhythm = {"event_type": event.event_type, "confidence": round(event.confidence, 3),
              "threshold": thr, "status": status}

    # --- vitals: every trigger, not just the first ---
    mews_thr = config.criticality_mews_threshold
    components = [{"name": c.name, "value": round_vital(c.name, c.value), "score": c.score}
                  for c in sorted(a.mews.components, key=lambda c: -c.score) if c.score > 0]
    deteriorating = []
    # most significant first, so a capped headline shows the strongest evidence
    for name, t in sorted(a.vital_trends.items(), key=lambda kv: (kv[1].p is None, kv[1].p or 0, kv[0])):
        if t.direction != "deteriorating":
            continue
        direction = "rising" if (t.slope or 0) > 0 else "falling" if (t.slope or 0) < 0 else "changing"
        deteriorating.append({"vital": name, "direction": direction, "p": t.p,
                              "samples": _trend_samples(event, name)})
    vitals = {
        "mews": {"score": a.mews.score, "risk": a.mews.risk, "threshold": mews_thr,
                 "triggered": a.mews.score >= mews_thr, "components": components},
        "deteriorating": deteriorating,
        "trend_triggered": bool(deteriorating) and config.criticality_escalate_on_deteriorating,
    }

    # --- criticality: the starting level and each trigger that raised it ---
    level = event_criticality(event, config)
    base = criticality(event.event_type, a.mews.risk)
    escalated_by = []
    if level != base:
        if not normal:
            escalated_by.append("non-normal rhythm")
        if vitals["mews"]["triggered"]:
            escalated_by.append(f"MEWS {a.mews.score} ≥ {mews_thr}")
        if config.criticality_escalate_on_deteriorating and is_deteriorating(a):
            escalated_by.append("deteriorating vitals")
    crit = {"level": level, "base": base, "escalated_by": escalated_by}

    out = {
        "decision": {"dispatch": dispatch, "reason_code": code, "reason": reason,
                     "summary": REASON_TEXT.get(code, code)},
        "basis": basis if dispatch else "none",
        "rhythm": rhythm, "vitals": vitals, "criticality": crit,
    }
    out["headline"] = headline(out)
    return out


def _trend_samples(event, vital: str) -> list[dict]:
    """The readings the trend was computed from: the window's history for `vital`, oldest first,
    at display precision. The vitals analysis runs Mann-Kendall on exactly these (sorted by time),
    so this is the evidence behind "rising"/"falling". Empty when the history isn't available
    (e.g. a payload that didn't carry it)."""
    history = getattr(getattr(event, "window", None), "vitals_history", None) or {}
    samples = sorted(history.get(vital, []), key=lambda s: s.timestamp)
    return [{"t": s.timestamp, "v": round_vital(vital, s.value)} for s in samples]


def _vitals_phrase(vitals: dict) -> str:
    """`MEWS 6 ≥ 3 (HR 2, resp rate 2); HR rising (p=0.01), SpO₂ falling (p=0.03)` or ''."""
    parts = []
    m = vitals["mews"]
    if m["triggered"]:
        comp = ", ".join(f"{c['name']} {c['score']}" for c in m["components"][:3])
        parts.append(f"MEWS {m['score']} ≥ {m['threshold']}" + (f" ({comp})" if comp else ""))
    det = vitals["deteriorating"]
    if det:
        trends = ", ".join(f"{_VITAL_NAMES.get(d['vital'], d['vital'])} {d['direction']}{_p(d['p'])}"
                           for d in det[:_MAX_TRENDS])
        if len(det) > _MAX_TRENDS:
            trends += f" (+{len(det) - _MAX_TRENDS} more)"
        parts.append(trends)
    return "; ".join(parts)


def headline(x: dict) -> str:
    """One sentence: why the event is shown (or why it was not alerted)."""
    r, crit, d = x["rhythm"], x["criticality"]["level"], x["decision"]
    name = _rhythm_name(r["event_type"])
    vit = _vitals_phrase(x["vitals"])
    if d["dispatch"]:
        if x["basis"] == "rhythm":
            s = f"Shown because: {name} ({_pct(r['confidence'])} ≥ {_pct(r['threshold'])}), {crit}"
            return s + (f"; vitals also abnormal: {vit}." if vit else ".")
        if r["status"] == "normal":
            return (f"Shown because the vitals warrant it, not the rhythm: {vit or 'vitals'}. "
                    f"Rhythm reads normal ({_pct(r['confidence'])}). {crit}.")
        return (f"Shown because the vitals warrant it: {vit or 'vitals'}. Rhythm {name} unconfirmed "
                f"({_pct(r['confidence'])} < {_pct(r['threshold'])}). {crit}.")
    code = d["reason_code"]
    if code == "false_positive":
        return f"Not alerted: rhythm reads normal ({_pct(r['confidence'])}) and the vitals are calm."
    if code == "low_confidence_arrhythmia":
        return (f"Not alerted: {name} only {_pct(r['confidence'])} (< {_pct(r['threshold'])}) "
                f"and the vitals are calm.")
    if code == "below_threshold":
        return f"Not alerted: {name}, criticality {crit} is below the alert threshold."
    return f"Not alerted: {d['reason']}."
