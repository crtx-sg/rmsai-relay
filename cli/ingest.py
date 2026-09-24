"""Phase 1 ingest CLI: HDF5 file -> reader -> model -> FP gate -> vitals -> DeviceEvent.

  python -m cli.ingest --file data/fixtures/PT1155_2026-06.h5 --emit stdout
  python -m cli.ingest --file PT1234_2025-09.h5 --emit bus --stream rmsai.events
  python -m cli.ingest --dir data/real --emit bus     # every *.h5 (e.g. from cli.real_samples)

`--emit stdout` prints a per-event summary + the markdown report. `--emit bus` publishes each
enriched `DeviceEvent` (raw signals excluded) to a Redis Stream for the consumer pool (§3.2).
When events carry a ground-truth class, a scored summary goes to stderr at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.audit import AuditLog
from common.config import DEFAULT
from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from common.preflight import service_unreachable
from inference.ecg_model import get_ecg_model
from inference.pipeline import process_window
from inference.serialize import event_summary_line, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file


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
    parser.add_argument("--show-report", action="store_true", help="Print markdown report (stdout).")
    args = parser.parse_args(argv)

    # `--checkpoint` with no values means "stub, ignore the env"; omitting it falls back to config.
    checkpoints = DEFAULT.ecg_checkpoints if args.checkpoint is None else args.checkpoint
    model = get_ecg_model(checkpoints)
    vitals = MewsVitalsAnalysis()
    audit = AuditLog(DEFAULT.audit_log_path)

    files = sorted(Path(args.dir).glob("*.h5")) if args.dir else [Path(args.file)]
    n = scored = correct = 0
    for window in (w for f in files for w in read_hdf5_file(f, strict_units=args.strict_units)):
        event = process_window(window, model, vitals)
        # Render the ECG strip here, while the raw samples are in hand (the bus drops them); the path
        # rides along in the payload and is persisted as MonitoredEvent.ecg_plot_ref downstream.
        if DEFAULT.ecg_plot_enabled:
            from inference.plotting import render_ecg_strip  # noqa: PLC0415

            event.window.ecg_plot_ref = render_ecg_strip(event.window, config=DEFAULT)
        n += 1
        truth = window.ground_truth.condition if window.ground_truth else None
        if truth in CLASS_NAMES:
            scored += 1
            correct += truth == event.event_type
        if args.emit == "bus":
            msg_id = publish_to_bus(args.redis_url, args.stream, event_to_dict(event))
            audit.write(actor="cli.ingest", action="emit_event", subject=window.patient_ref,
                        outcome="published", stream=args.stream, msg_id=msg_id)
            print(json.dumps({"published": msg_id, **event_summary_line(event)}))
        else:
            print(json.dumps(event_summary_line(event)))
            if args.show_report:
                print(event.report_md)

    if n == 0:
        print("no readable events", file=sys.stderr)
        return 1
    if scored:
        stub = isinstance(model, StubECGModel)
        print(json.dumps({"summary": {"events": n, "scored": scored, "correct": correct,
                                      "accuracy": round(correct / scored, 3),
                                      "model": "stub" if stub else "checkpoint"}}), file=sys.stderr)
        if stub:
            print("[ingest] WARNING scored against the deterministic STUB — set ECG_CHECKPOINTS "
                  "(or --checkpoint) for real predictions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
