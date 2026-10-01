"""One readable log line per event, and a rolling performance summary (pure; used by cli.consume).

The line answers, at a glance: whose event, where the data came from, what the model said vs. the
truth (and whether that was right), how critical, whether an alert went out and through what, and
why. For example:

    [event] PT992591 · INCART I05 @263874 (test split) · pred VENTRICULAR_TACHYCARDIA 50% ·
            truth VENTRICULAR_TACHYCARDIA → TP · Critical · alert ✓ vitals_alert · app ✗ ·
            why: Shown because the vitals warrant it: …

Patients appear by pseudonym only (rule 6). `PerfTracker` keeps the labelled events seen by this
process and prints `inference.metrics.format_summary` every N of them, one block per model, since
mixing models would make the numbers meaningless.
"""

from __future__ import annotations

from collections.abc import Callable

from common.config import DEFAULT, Config
from inference.metrics import EvalRecord, classify_outcome, format_summary, summarize

DATASET_NAMES = {"mitbih": "MIT-BIH", "incart": "INCART", "ptbxl": "PTB-XL", "vfdb": "VFDB",
                 "afdb": "AFDB", "cudb": "CUDB"}


def _get(p, key):
    return p.get(key) if isinstance(p, dict) else getattr(p, key, None)


def source_label(provenance) -> str:
    """`MIT-BIH 207 @221697 (test split)` | `simulator` | `device <id>` | `source unknown`.

    Accepts a `Provenance` or the equivalent dict (as stored on the graph or carried in a trace).
    """
    if provenance is None:
        return "source unknown"
    kind = _get(provenance, "kind")
    if kind == "simulator":
        return "simulator"
    if kind == "device":
        return f"device {_get(provenance, 'device') or '?'}"
    dataset = _get(provenance, "dataset") or "?"
    record = str(_get(provenance, "record") or "?")
    record = record.split(":", 1)[1] if ":" in record else record  # "incart:I05" → "I05"
    out = f"{DATASET_NAMES.get(dataset, dataset.upper())} {record}"
    if _get(provenance, "source_sample") is not None:
        out += f" @{_get(provenance, 'source_sample')}"
    if _get(provenance, "split"):
        out += f" ({_get(provenance, 'split')} split)"
    return out


def event_trace(event, config: Config = DEFAULT) -> dict:
    """The per-event facts the log line and the tracker need, computed once."""
    from orchestrator.explain import explain_event  # noqa: PLC0415

    w = event.window
    x = explain_event(event, config)
    truth = w.ground_truth.condition if w.ground_truth else None
    outcome = classify_outcome(event.event_type, truth, event.model_classes)
    return {
        "patient": w.patient_ref, "event_id": w.event_id,
        "source": source_label(w.provenance),
        "source_kind": w.provenance.kind if w.provenance else None,
        "dataset": w.provenance.dataset if w.provenance else None,
        "predicted": event.event_type, "confidence": event.confidence, "truth": truth,
        "outcome": outcome.code, "model_id": event.model_id, "model_classes": event.model_classes,
        "criticality": x["criticality"]["level"],
        "gate": x["decision"]["dispatch"], "reason_code": x["decision"]["reason_code"],
        "why": x["headline"],
    }


def event_log_line(trace: dict, *, app: bool | None = None, call: str | None = None,
                   sms: str | None = None) -> str:
    """The `[event] …` line. Delivery parts appear only when they apply."""
    parts = [
        trace["patient"], trace["source"],
        f"pred {trace['predicted']} {trace['confidence']:.0%}",
        (f"truth {trace['truth']} → {trace['outcome']}" if trace["truth"] else "truth —"),
        trace["criticality"],
        f"alert {'✓' if trace['gate'] else '✗'} {trace['reason_code']}",
    ]
    if trace["gate"] and app is not None:
        parts.append(f"app {'✓' if app else '✗'}")
    if call:
        parts.append(f"call {call}")
    if sms:
        parts.append(sms.replace("_", " "))
    return "[event] " + " · ".join(parts) + f" · why: {trace['why']}"


class PerfTracker:
    """Labelled events seen by this process; prints a per-model summary every `every` of them."""

    def __init__(self, every: int = 10, emit: Callable[[str], None] = print) -> None:
        self.every = every
        self.emit = emit
        self._by_model: dict[str, dict] = {}  # model_id -> {"records": [...], "classes": [...]}
        self._labelled = 0

    def add(self, trace: dict) -> None:
        if not trace.get("truth"):
            return
        m = self._by_model.setdefault(trace.get("model_id") or "unknown model",
                                      {"records": [], "classes": trace.get("model_classes")})
        m["records"].append(EvalRecord(trace["predicted"], trace["truth"], trace["confidence"],
                                       dispatched=trace["gate"]))
        self._labelled += 1
        if self.every and self._labelled % self.every == 0:
            self.flush(header=f"after {self._labelled} labelled events")

    def flush(self, header: str = "summary") -> None:
        """Print one block per model (nothing if no labelled events were seen)."""
        for model, m in self._by_model.items():
            self.emit(f"[perf] {header} · model {model}\n"
                      + format_summary(summarize(m["records"], m["classes"])))
