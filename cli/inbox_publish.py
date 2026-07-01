"""Phase 9 harness: publish a critical-event worklist notification into the hospital inbox.

Demonstrates the app-surface push in isolation — mint scoped artifact links, build the pseudonym-only
worklist message, publish it to `rmsai-inbox-<hospital_id>`, and round-trip a token through the store.

  # offline, no infra — print the message + a mint/verify/expiry roundtrip
  python -m cli.inbox_publish --dry-run

  # real push (needs Redis + LIVEKIT_* configured); the app, joined to the inbox room, sees it
  python -m cli.inbox_publish --hospital-id h1 --event-type ATRIAL_FIBRILLATION
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from common.config import Config
from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from live.artifact_tokens import ArtifactTokenStore
from live.inbox import (
    InboxPublisher,
    artifact_kinds_for,
    build_event_message,
    inbox_room,
    mint_artifact_links,
)

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _StubModel(ECGModel):
    def __init__(self, event_type: str, confidence: float = 0.92) -> None:
        self.event_type = event_type
        self.confidence = confidence

    def predict(self, window):
        return self.event_type, self.confidence


class _MemRedis:
    """Tiny in-memory Redis stand-in (set/get/delete + TTL) for `--dry-run`, honoring a virtual clock."""

    def __init__(self, clock: float | None = None) -> None:
        self._d: dict[str, tuple[str, float | None]] = {}
        self.clock = time.time() if clock is None else clock

    def set(self, key: str, val: str, ex: int | None = None) -> None:
        self._d[key] = (val, None if ex is None else self.clock + ex)

    def get(self, key: str):
        item = self._d.get(key)
        if item is None:
            return None
        val, exp = item
        if exp is not None and self.clock >= exp:
            self._d.pop(key, None)
            return None
        return val.encode("utf-8")

    def delete(self, key: str) -> None:
        self._d.pop(key, None)


def _build_event(event_type: str, hr: float):
    w = next(read_hdf5_file(_FIXTURE))
    if "HR" in w.vitals:
        w.vitals["HR"].value = hr
    return process_window(w, _StubModel(event_type), MewsVitalsAnalysis())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hospital-id", default=None, help="override HOSPITAL_ID for the inbox room")
    parser.add_argument("--event-type", default="ATRIAL_FIBRILLATION")
    parser.add_argument("--hr", type=float, default=145.0, help="HR to force a high-MEWS critical event")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the message + a token roundtrip; no Redis/LiveKit needed")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if args.hospital_id is not None:
        config = replace(config, hospital_id=args.hospital_id)

    event = _build_event(args.event_type, args.hr)
    w = event.window
    kinds = artifact_kinds_for(event, config=config)

    if args.dry_run:
        store = ArtifactTokenStore(_MemRedis(), ttl_seconds=300)
        publisher = InboxPublisher(
            inbox_room(config),
            send_fn=lambda room, data: print(f"[dry-run] send_data -> {room}:\n"
                                             f"{json.dumps(json.loads(data), indent=2)}"),
        )
    else:
        store = ArtifactTokenStore.from_config(config)
        publisher = InboxPublisher.from_config(config)

    links = mint_artifact_links(w.event_id, kinds, store)
    message = build_event_message(
        event_id=w.event_id, patient=w.patient_ref, unit="ICU", bed="3",
        event_type=event.event_type, ts=w.event_timestamp, criticality="High",
        status="reported", links=links,
    )
    publisher.publish_event(message)

    # Token roundtrip: a valid link resolves; an unknown one is refused.
    for kind, link in links.items():
        grant = store.verify(link["token"], kind)
        print(f"[verify] {kind}: token -> {grant}")
    print(f"[verify] bogus token -> {store.verify('nope', 'report')}")

    # Status roundtrip (what an acknowledge would push back).
    publisher.publish_status(w.event_id, "acknowledged")
    print(f"[ok] published event + status for {w.event_id} to {publisher.room}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
