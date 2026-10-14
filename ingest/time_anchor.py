"""Re-anchor a recording's timestamps so its last event is "now" (demo time).

ecg_sigma gives datasets without absolute time (MIT-BIH, AFDB, VFDB, INCART) a synthetic date:
2025-01-01 + a per-record offset of up to 28 days + the event's position in the recording. PTB-XL
keeps its real 1995-96 acquisition dates. Neither reads as "now" on a worklist, and "events in the
last N hours" never matches. `rebase_by_patient` shifts every timestamp of a patient's recordings
by the same amount, so their last event lands at `now` and all spacing (between events, between
recordings, and within each vital's history, which MEWS trends use) is exactly as recorded. The original time is not lost: `cli.ingest` reports it as `source_ts` in its output and audit entry.
"""

from __future__ import annotations

from common.schemas import SignalWindow

ANCHORS = ("source", "now")


def _shift_window(w: SignalWindow, delta: float) -> SignalWindow:
    return w.model_copy(update={
        "event_timestamp": w.event_timestamp + delta,
        "start_timestamp": w.start_timestamp + delta,
        "vitals": {k: v.model_copy(update={"timestamp": v.timestamp + delta})
                   for k, v in w.vitals.items()},
        "vitals_history": {k: [s.model_copy(update={"timestamp": s.timestamp + delta}) for s in h]
                           for k, h in w.vitals_history.items()},
    })


def rebase_by_patient(windows: list[SignalWindow], now: float) -> list[SignalWindow]:
    """Shift each PATIENT's windows (across all their recordings) so their latest event is `now`.

    Per-file anchoring put two recordings of one patient (e.g. INCART I04 and I05, one subject)
    both at "now", erasing the week between them; per patient, that gap is kept. Order unchanged.
    """
    latest: dict[str, float] = {}
    for w in windows:
        latest[w.patient_ref] = max(latest.get(w.patient_ref, w.event_timestamp), w.event_timestamp)
    return [_shift_window(w, now - latest[w.patient_ref]) for w in windows]

