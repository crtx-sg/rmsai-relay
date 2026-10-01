"""Model-performance scoring against ground truth: per-event outcome + summary metrics (pure).

Two levels, both binary with **positive = any arrhythmia** (anything but `NORMAL_SINUS`):

* **model level:** what the classifier said vs. the true label;
    - TP: arrhythmia predicted for an arrhythmia. `class_match` says whether it was the *right*
      arrhythmia; a wrong one (VT read as VF) is "TP, wrong class": the alarm was right, the class
      wasn't.
    - FP: arrhythmia predicted for a normal rhythm.
    - FN: normal predicted for an arrhythmia.
    - TN: normal for normal.
* **alert level:** whether the event was *dispatched* (after the criticality gate) vs. the true
  label: correct alert, missed alert, false alert, correctly silent. This is the clinically
  meaningful level, since vitals can raise an alert the rhythm doesn't support, and the
  confidence gate can withhold one.

An event is **unscorable** without a ground truth, or when its true label isn't a class the model
can output (e.g. `OTHER`, or `ST_ELEVATION` for the 13-class `real_v2` head). Unscorable events are
counted and reported, never scored as right or wrong.

Rates come with Wilson 95 % intervals and their raw counts, because demo sets are small and a bare
"100 %" over 5 events claims far more than it shows.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from common.event_types import CLASS_NAMES

NORMAL = "NORMAL_SINUS"
_Z95 = 1.959963984540054

# Model-level outcome codes; alert-level outcome codes.
TP, FP, FN, TN, UNSCORABLE = "TP", "FP", "FN", "TN", "UNSCORABLE"
ALERT_KINDS = ("alert_correct", "alert_missed", "alert_false", "silent_correct")


@dataclass(frozen=True)
class Outcome:
    kind: str                      # TP | FP | FN | TN | UNSCORABLE
    class_match: bool | None = None  # exact class right? (None when unscorable)
    reason: str = ""                 # why unscorable

    @property
    def code(self) -> str:
        """Machine form for JSON/logs: TP | TP_WRONG_CLASS | FP | FN | TN | UNSCORABLE."""
        if self.kind == TP and not self.class_match:
            return "TP_WRONG_CLASS"
        return self.kind

    @property
    def label(self) -> str:
        """Short display form: `TP ✓`, `TP ✗class`, `FP`, `FN`, `TN`, `—`."""
        if self.kind == UNSCORABLE:
            return "—"
        if self.kind == TP:
            return "TP ✓" if self.class_match else "TP ✗class"
        return self.kind


@dataclass(frozen=True)
class EvalRecord:
    """One event's facts for scoring. `dispatched` is None when the alert decision is unknown."""

    predicted: str
    truth: str | None
    confidence: float | None = None
    dispatched: bool | None = None


def classify_outcome(predicted: str, truth: str | None,
                     model_classes: Sequence[str] | None = None, *, normal: str = NORMAL) -> Outcome:
    """Model-level outcome of one event (see the module docstring)."""
    if not truth:
        return Outcome(UNSCORABLE, reason="no ground truth")
    if truth not in CLASS_NAMES:
        return Outcome(UNSCORABLE, reason=f"label {truth} is not a model class")
    if model_classes is not None and truth not in model_classes:
        return Outcome(UNSCORABLE, reason=f"label {truth} is outside this model's "
                                          f"{len(model_classes)} classes")
    pred_pos, true_pos = predicted != normal, truth != normal
    if pred_pos and true_pos:
        return Outcome(TP, class_match=predicted == truth)
    if pred_pos:
        return Outcome(FP, class_match=False)
    if true_pos:
        return Outcome(FN, class_match=False)
    return Outcome(TN, class_match=True)


def classify_alert(dispatched: bool, truth: str | None,
                   model_classes: Sequence[str] | None = None, *, normal: str = NORMAL) -> str | None:
    """Alert-level outcome: `alert_correct` | `alert_missed` | `alert_false` | `silent_correct`.

    None when the event is unscorable (same rules as `classify_outcome`).
    """
    if classify_outcome(normal, truth, model_classes, normal=normal).kind == UNSCORABLE:
        return None
    true_pos = truth != normal
    if dispatched:
        return "alert_correct" if true_pos else "alert_false"
    return "alert_missed" if true_pos else "silent_correct"


def wilson(k: int, n: int) -> dict | None:
    """`{value, low, high, k, n}` for a proportion k/n with a Wilson 95 % interval; None if n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / denom
    return {"value": round(p, 3), "low": round(max(0.0, centre - half), 3),
            "high": round(min(1.0, centre + half), 3), "k": k, "n": n}


def _f1(precision: dict | None, recall: dict | None) -> float | None:
    if not precision or not recall or precision["value"] + recall["value"] == 0:
        return None
    p, r = precision["value"], recall["value"]
    return round(2 * p * r / (p + r), 3)


def summarize(records: Iterable[EvalRecord], model_classes: Sequence[str] | None = None) -> dict:
    """Summary metrics over `records` (model level, per class, confusion, alert level)."""
    records = list(records)
    counts: Counter = Counter()
    unscorable: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    conf_right: list[float] = []
    conf_wrong: list[float] = []
    alert_counts: Counter = Counter()
    exact = 0

    for r in records:
        o = classify_outcome(r.predicted, r.truth, model_classes)
        if o.kind == UNSCORABLE:
            unscorable[o.reason] += 1
            continue
        counts[o.kind] += 1
        if o.kind == TP and not o.class_match:
            counts["TP_wrong_class"] += 1
        if o.class_match:
            exact += 1
        confusion[r.truth][r.predicted] += 1
        if r.confidence is not None:
            (conf_right if o.class_match else conf_wrong).append(r.confidence)
        if r.dispatched is not None:
            alert_counts[classify_alert(r.dispatched, r.truth, model_classes)] += 1

    tp, fp, fn, tn = (counts[k] for k in (TP, FP, FN, TN))
    scored = tp + fp + fn + tn
    sens, spec = wilson(tp, tp + fn), wilson(tn, tn + fp)
    ppv, npv = wilson(tp, tp + fp), wilson(tn, tn + fn)

    per_class = {}
    for cls in sorted({c for row in confusion for c in (row, *confusion[row])}):
        row = confusion.get(cls, {})  # .get: a class only ever *predicted* has no truth row
        c_tp = row.get(cls, 0)
        c_fp = sum(confusion[t].get(cls, 0) for t in confusion if t != cls)
        c_fn = sum(n for p, n in row.items() if p != cls)
        precision, recall = wilson(c_tp, c_tp + c_fp), wilson(c_tp, c_tp + c_fn)
        per_class[cls] = {"support": c_tp + c_fn, "tp": c_tp, "fp": c_fp, "fn": c_fn,
                          "precision": precision, "recall": recall, "f1": _f1(precision, recall)}
    f1s = [v["f1"] or 0.0 for v in per_class.values() if v["support"] > 0]

    out = {
        "events": len(records), "scored": scored,
        "unscorable": sum(unscorable.values()), "unscorable_reasons": dict(unscorable),
        "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "TP_wrong_class": counts["TP_wrong_class"]},
        "accuracy_exact": wilson(exact, scored),
        "accuracy_binary": wilson(tp + tn, scored),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "f1_binary": _f1(ppv, sens),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else None,
        "per_class": per_class,
        "confusion": {t: dict(p) for t, p in sorted(confusion.items())},
        "mean_confidence": {
            "correct": round(sum(conf_right) / len(conf_right), 3) if conf_right else None,
            "wrong": round(sum(conf_wrong) / len(conf_wrong), 3) if conf_wrong else None,
        },
    }
    if alert_counts:
        a = {k: alert_counts[k] for k in ALERT_KINDS}
        out["alert"] = {
            "counts": a,
            "sensitivity": wilson(a["alert_correct"], a["alert_correct"] + a["alert_missed"]),
            "false_alert_rate": wilson(a["alert_false"], a["alert_false"] + a["silent_correct"]),
            "ppv": wilson(a["alert_correct"], a["alert_correct"] + a["alert_false"]),
        }
    return out


def _pct(m: dict | None) -> str:
    if not m:
        return "n/a"
    return f"{m['value']:.0%} ({m['k']}/{m['n']}, 95% CI {m['low']:.0%}–{m['high']:.0%})"


def _num(x) -> str:
    return "-" if x is None else str(x)


def format_summary(s: dict) -> str:
    """Human-readable summary block for the CLI and logs."""
    c = s["counts"]
    lines = [
        f"Model performance — {s['scored']} scored of {s['events']} events"
        + (f" ({s['unscorable']} unscorable: "
           + ", ".join(f"{n}× {why}" for why, n in s["unscorable_reasons"].items()) + ")"
           if s["unscorable"] else ""),
        f"  outcomes     TP {c['TP']} (wrong class {c['TP_wrong_class']}) · FP {c['FP']} · "
        f"FN {c['FN']} · TN {c['TN']}",
        f"  exact class  {_pct(s['accuracy_exact'])}",
        f"  sensitivity  {_pct(s['sensitivity'])}",
        f"  specificity  {_pct(s['specificity'])}",
        f"  PPV          {_pct(s['ppv'])}",
        f"  NPV          {_pct(s['npv'])}",
        f"  F1 (binary) {_num(s['f1_binary'])} · macro-F1 (per class) {_num(s['macro_f1'])}",
    ]
    mc = s["mean_confidence"]
    if mc["correct"] is not None or mc["wrong"] is not None:
        lines.append(f"  mean confidence  correct {mc['correct']} · wrong {mc['wrong']}")
    if s["per_class"]:
        lines.append("  per class    support  precision  recall  f1")
        for cls, v in s["per_class"].items():
            prec = f"{v['precision']['value']:.2f}" if v["precision"] else "  - "
            rec = f"{v['recall']['value']:.2f}" if v["recall"] else "  - "
            f1 = v["f1"] if v["f1"] is not None else "-"
            lines.append(f"    {cls:<26}{v['support']:>4}     {prec:>5}    {rec:>5}  {f1}")
    if "alert" in s:
        a = s["alert"]
        lines.append(f"  alerts       correct {a['counts']['alert_correct']} · missed "
                     f"{a['counts']['alert_missed']} · false {a['counts']['alert_false']} · "
                     f"silent-correct {a['counts']['silent_correct']}")
        lines.append(f"    alert sensitivity {_pct(a['sensitivity'])}")
        lines.append(f"    false-alert rate  {_pct(a['false_alert_rate'])}")
    return "\n".join(lines)
