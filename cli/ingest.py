"""Phase 1 ingest CLI: HDF5 file -> reader -> model -> FP gate -> vitals -> DeviceEvent.

  python -m cli.ingest --file data/fixtures/PT1155_2026-06.h5 --emit stdout
  python -m cli.ingest --file PT1234_2025-09.h5 --emit bus --stream rmsai.events
  python -m cli.ingest --dir data/real --emit bus     # every *.h5 (e.g. from cli.real_samples)

`--emit stdout` prints a per-event summary + the markdown report. `--emit bus` publishes each
enriched `DeviceEvent` (raw signals excluded) to a Redis Stream for the consumer pool (§3.2).
When events carry a ground-truth class, each line gains an `outcome` (TP / TP_WRONG_CLASS / FP / FN /
TN / UNSCORABLE) and a scored summary goes to stderr at the end; `--metrics` prints the
full breakdown (sensitivity/specificity/PPV/NPV with 95% CIs, per class, alert level).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from common.audit import AuditLog
from common.config import DEFAULT
from common.ecg_model_stub import StubECGModel
from common.preflight import service_unreachable
from inference.ecg_model import get_ecg_model
from inference.metrics import EvalRecord, classify_outcome, format_summary, summarize
from inference.pipeline import process_window
from inference.serialize import event_summary_line, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from ingest.time_anchor import ANCHORS, rebase_by_patient


def publish_to_bus(redis_url: str, stream: str, payload: dict) -> str:
    import redis  # noqa: PLC0415

    client = redis.Redis.from_url(redis_url)
    fields = {
        "data": json.dumps(payload),
        "patient": payload["patient_ref"],
        "uuid": payload["event_id"],
        "event_type": payload["event_type"],
        "criticality": payload["criticality"],
        "is_false_positive": str(payload["is_false_positive"]),
    }
    try:
        return client.xadd(stream, fields).decode()
    except redis.exceptions.ConnectionError as exc:
        # `from None`: the pool's traceback is noise once the cause has a name and a fix.
        raise service_unreachable("Redis (the event bus)", redis_url, exc) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="HDF5 archive to ingest.")
    src.add_argument("--dir", help="ingest every *.h5 in this directory (sorted).")
    parser.add_argument("--emit", choices=["stdout", "bus"], default="stdout")
    parser.add_argument("--stream", default="rmsai.events", help="Redis Stream name (--emit bus).")
    parser.add_argument("--redis-url", default=DEFAULT.redis_url)
    parser.add_argument(
        "--checkpoint", nargs="*", default=None,
        help="ECG model checkpoint(s) (.pt). Several load as one softmax-averaging ensemble. "
             "Defaults to ECG_CHECKPOINTS; with neither, the deterministic stub is used.",
    )
    parser.add_argument("--strict-units", action="store_true", help="Fail if waveform_units absent.")
    parser.add_argument("--time-anchor", choices=ANCHORS, default=None,
                        help="source = HDF5 event times; now = shift each patient's events so their "
                             "latest is the ingest time (default INGEST_TIME_ANCHOR, else source).")
    parser.add_argument("--show-report", action="store_true", help="Print markdown report (stdout).")
    parser.add_argument("--explain", action="store_true",
                        help="Add `why` to each line: the headline of why the event is (or isn't) "
                             "alerted (orchestrator.explain).")
    parser.add_argument("--trend-samples", action="store_true",
                        help="With --explain: add `trend_samples`, the readings behind each "
                             "deteriorating trend.")
    parser.add_argument("--metrics", action="store_true",
                        help="Print the full model-performance breakdown (stderr) when events are "
                             "labelled.")
    args = parser.parse_args(argv)

    # `--checkpoint` with no values means "stub, ignore the env"; omitting it falls back to config.
    checkpoints = DEFAULT.ecg_checkpoints if args.checkpoint is None else args.checkpoint
    model = get_ecg_model(checkpoints)
    vitals = MewsVitalsAnalysis()
    audit = AuditLog(DEFAULT.audit_log_path)

    files = sorted(Path(args.dir).glob("*.h5")) if args.dir else [Path(args.file)]
    # The head the outcome is scored against: a real checkpoint's classes; the stub's are all of them.
    model_classes = getattr(model, "labels", None)
    # The alert gate exactly as cli.consume applies it (it forces outbound on), so the alert-level
    # metrics here predict what the consumer will dispatch.
    from orchestrator.outbound_flow import should_call  # noqa: PLC0415

    gate_config = replace(DEFAULT, outbound_enabled=True)
    n = 0
    records: list[EvalRecord] = []
    anchor = args.time_anchor or DEFAULT.ingest_time_anchor
    if anchor not in ANCHORS:
        parser.error(f"INGEST_TIME_ANCHOR must be one of {', '.join(ANCHORS)}, got {anchor!r}")
    now = time.time()

    def _windows():
        # Per-patient anchoring needs every recording first: a patient may span several files.
        source = [w for f in files for w in read_hdf5_file(f, strict_units=args.strict_units)]
        shifted = rebase_by_patient(source, now) if anchor == "now" else source
        yield from zip(source, shifted)

    for source_window, window in _windows():
        source_ts = source_window.event_timestamp
        event = process_window(window, model, vitals)
        # Render the ECG strip here, while the raw samples are in hand (the bus drops them); the path
        # rides along in the payload and is persisted as MonitoredEvent.ecg_plot_ref downstream.
        if DEFAULT.ecg_plot_enabled:
            from inference.plotting import render_ecg_strip  # noqa: PLC0415

            event.window.ecg_plot_ref = render_ecg_strip(event.window, config=DEFAULT)
        n += 1
        truth = window.ground_truth.condition if window.ground_truth else None
        would_alert, _ = should_call(event, gate_config)
        records.append(EvalRecord(event.event_type, truth, event.confidence, dispatched=would_alert))
        outcome = classify_outcome(event.event_type, truth, model_classes).code
        extra = {"outcome": outcome}
        if anchor == "now":
            extra["source_ts"] = source_ts  # the HDF5 event time, before re-anchoring
        if args.explain:
            from orchestrator.explain import explain_event  # noqa: PLC0415

            x = explain_event(event, gate_config)
            extra["why"] = x["headline"]
            if args.trend_samples:
                extra["trend_samples"] = {d["vital"]: [s["v"] for s in d["samples"]]
                                          for d in x["vitals"]["deteriorating"]}
        if args.emit == "bus":
            msg_id = publish_to_bus(args.redis_url, args.stream, event_to_dict(event))
            audit.write(actor="cli.ingest", action="emit_event", subject=window.patient_ref,
                        outcome="published", stream=args.stream, msg_id=msg_id,
                        **({"source_ts": source_ts} if anchor == "now" else {}))
            print(json.dumps({"published": msg_id, **event_summary_line(event), **extra},
                             ensure_ascii=False))
        else:
            print(json.dumps({**event_summary_line(event), **extra}, ensure_ascii=False))
            if args.show_report:
                print(event.report_md)

    if n == 0:
        print("no readable events", file=sys.stderr)
        return 1
    summary = summarize(records, model_classes)
    if summary["scored"]:
        stub = isinstance(model, StubECGModel)
        val = lambda m: m["value"] if m else None  # noqa: E731
        c = summary["counts"]
        print(json.dumps({"summary": {
            "events": n, "scored": summary["scored"], "unscorable": summary["unscorable"],
            "correct": summary["accuracy_exact"]["k"], "accuracy": val(summary["accuracy_exact"]),
            "tp": c["TP"], "fp": c["FP"], "fn": c["FN"], "tn": c["TN"],
            "sensitivity": val(summary["sensitivity"]), "specificity": val(summary["specificity"]),
            "alert_sensitivity": val(summary.get("alert", {}).get("sensitivity")),
            "false_alert_rate": val(summary.get("alert", {}).get("false_alert_rate")),
            "model": "stub" if stub else "checkpoint",
        }}), file=sys.stderr)
        if args.metrics:
            print(format_summary(summary), file=sys.stderr)
        if stub:
            print("[ingest] WARNING scored against the deterministic STUB — set ECG_CHECKPOINTS "
                  "(or --checkpoint) for real predictions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
