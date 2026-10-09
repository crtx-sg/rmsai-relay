"""Re-anchor a recording's timestamps so its last event is "now" (demo time).

ecg_sigma gives datasets without absolute time (MIT-BIH, AFDB, VFDB, INCART) a synthetic date:
2025-01-01 + a per-record offset of up to 28 days + the event's position in the recording. PTB-XL
keeps its real 1995-96 acquisition dates. Neither reads as "now" on a worklist, and "events in the
last N hours" never matches. `rebase_to_now` shifts every timestamp of one recording by the same
amount, so the last event lands at `now` and all spacing (between events, and within each vital's
history, which MEWS trends use) is exactly as recorded.

The original time is not lost: `cli.ingest` reports it as `source_ts` in its output and audit entry.
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


def rebase_to_now(windows: list[SignalWindow], now: float) -> list[SignalWindow]:
    """One recording's windows, shifted so the latest event is at `now` (order unchanged)."""
    if not windows:
        return []
    delta = now - max(w.event_timestamp for w in windows)
    return [_shift_window(w, delta) for w in windows]
