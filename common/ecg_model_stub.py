"""Deterministic `ECGModel` stub — used until real checkpoints are present.

Maps a `SignalWindow` to a stable `(event_type, confidence)` by hashing the signal content, so
the same window always yields the same prediction (test discipline, hard rule 7). No torch, no
weights. The real wrapper over the `ecgtranscnn` classifier lands in `inference/` (Phase 1).
"""

from __future__ import annotations

import hashlib
import struct

from .event_types import CLASS_NAMES
from .interfaces import ECGModel
from .schemas import SignalWindow


def _digest(window: SignalWindow) -> bytes:
    h = hashlib.sha256()
    h.update(window.event_id.encode("utf-8"))
    for name in sorted(window.signals):
        h.update(name.encode("utf-8"))
        # Hash a bounded prefix so large windows stay cheap but content-sensitive.
        for sample in window.signals[name][:64]:
            h.update(struct.pack("<d", float(sample)))
    return h.digest()


class StubECGModel(ECGModel):
    """Content-addressed deterministic classifier."""

    def predict(self, window: SignalWindow) -> tuple[str, float]:
        d = _digest(window)
        idx = d[0] % len(CLASS_NAMES)
        # Confidence in [0.55, 0.99], deterministic from the digest.
        confidence = 0.55 + (d[1] / 255.0) * 0.44
        return CLASS_NAMES[idx], round(confidence, 4)
