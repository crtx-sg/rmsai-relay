"""Phase 1 per-event pipeline: `SignalWindow` -> enriched `DeviceEvent`.

Stages (sequential within an event):
  1. ECGModel.predict  -> event_type + confidence
  2. false-positive gate + confidence thresholds (G7)
  3. VitalsAnalysis.analyze -> MEWS + trends + correlation notes
  4. render per-event markdown report (D18)
"""

from __future__ import annotations

from common.config import DEFAULT, Config
from common.event_types import NORMAL_SINUS
from common.interfaces import ECGModel, VitalsAnalysis
from common.schemas import DeviceEvent, SignalWindow

from .report import render_event_report


def process_window(
    window: SignalWindow,
    model: ECGModel,
    vitals: VitalsAnalysis,
    config: Config = DEFAULT,
) -> DeviceEvent:
    """Run the full inference + analysis pipeline for one event."""
    event_type, confidence = model.predict(window)

    # --- False-positive gate + confidence thresholds (G7) ---
    is_normal = event_type == NORMAL_SINUS
    is_false_positive = is_normal and confidence >= config.fp_suppress_min_confidence
    # NORMAL_SINUS predicted but below the suppression threshold => don't suppress; flag uncertain.
    uncertain = is_normal and not is_false_positive
    low_confidence = confidence < config.low_confidence_caveat

    analysis = vitals.analyze(window, event_type=event_type)

    event = DeviceEvent(
        window=window,
        event_type=event_type,
        confidence=confidence,
        is_false_positive=is_false_positive,
        uncertain=uncertain,
        low_confidence=low_confidence,
        analysis=analysis,
    )
    event.report_md = render_event_report(event)
    return event
