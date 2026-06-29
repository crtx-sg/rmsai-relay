"""Phase 1 ingest CLI: HDF5 file -> reader -> model -> FP gate -> vitals -> DeviceEvent.

  python -m cli.ingest --file data/fixtures/PT1155_2026-06.h5 --emit stdout
  python -m cli.ingest --file PT1234_2025-09.h5 --emit bus --stream rmsai.events

`--emit stdout` prints a per-event summary + the markdown report. `--emit bus` publishes each
enriched `DeviceEvent` (raw signals excluded) to a Redis Stream for the consumer pool (§3.2).
"""

from __future__ import annotations

import argparse
import json
import sys

from common.audit import AuditLog
from common.config import DEFAULT
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
    return client.xadd(stream, fields).decode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="HDF5 archive to ingest.")
    parser.add_argument("--emit", choices=["stdout", "bus"], default="stdout")
    parser.add_argument("--stream", default="rmsai.events", help="Redis Stream name (--emit bus).")
    parser.add_argument("--redis-url", default=DEFAULT.redis_url)
    parser.add_argument("--checkpoint", default=None, help="ECG model checkpoint (.pt); else stub.")
    parser.add_argument("--strict-units", action="store_true", help="Fail if waveform_units absent.")
    parser.add_argument("--show-report", action="store_true", help="Print markdown report (stdout).")
    args = parser.parse_args(argv)

    model = get_ecg_model(args.checkpoint)
    vitals = MewsVitalsAnalysis()
    audit = AuditLog(DEFAULT.audit_log_path)

    n = 0
    for window in read_hdf5_file(args.file, strict_units=args.strict_units):
        event = process_window(window, model, vitals)
        # Render the ECG strip here, while the raw samples are in hand (the bus drops them); the path
        # rides along in the payload and is persisted as MonitoredEvent.ecg_plot_ref downstream.
        if DEFAULT.ecg_plot_enabled:
            from inference.plotting import render_ecg_strip  # noqa: PLC0415

            event.window.ecg_plot_ref = render_ecg_strip(event.window, config=DEFAULT)
        n += 1
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
