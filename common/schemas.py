"""Frozen data contracts — the seams between subsystems.

These are deliberately decoupled from the external HDF5 layout (Appendix A). The `ingest/`
reader maps whatever file/stream format arrives onto `SignalWindow`; a format change is a
one-file edit in `ingest/`, never a change here.

Key invariant: **`SignalWindow` carries no predicted `event_type`.** The model predicts it in
the inference stage, producing a `DeviceEvent`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .event_types import CLASS_NAMES

# --------------------------------------------------------------------------------------
# Telemetry plane
# --------------------------------------------------------------------------------------


class Vital(BaseModel):
    """A single point vital measurement with its own units + timestamp."""

    value: float
    units: str
    timestamp: float  # epoch seconds


class VitalSample(BaseModel):
    """One historical vital reading (oldest-first in a history list)."""

    value: float
    timestamp: float


class WindowGeometry(BaseModel):
    """Window timing. `alarm_offset_seconds == before_s` always (decision G)."""

    before_s: float
    after_s: float
    sample_counts: dict[str, int] = Field(default_factory=dict)  # group -> n samples


class GroundTruth(BaseModel):
    """Simulator-only label — for evaluation/match, never used as the event_type."""

    condition: str
    heart_rate: Optional[float] = None
    event_timestamp: Optional[float] = None


class SignalWindow(BaseModel):
    """Reader output: raw signals + vitals + geometry. **No predicted event_type.**"""

    patient_ref: str  # pseudonym; maps directly to Patient.id (decision F)
    event_id: str  # event_<id>/uuid
    start_timestamp: float  # derived: event timestamp - before_s (decision B)
    event_timestamp: float  # authoritative per-event epoch

    # Multi-rate signals, in mV (decision C). lead/channel name -> samples.
    signals: dict[str, list[float]] = Field(default_factory=dict)
    # group (ecg/ppg/resp) -> rational sample rate (RESP = 100/3, not 33.33) (decision E).
    sample_rates: dict[str, Fraction] = Field(default_factory=dict)
    waveform_units: str = "mV"  # from file-level metadata (decision C)

    window: WindowGeometry
    vitals: dict[str, Vital] = Field(default_factory=dict)
    vitals_history: dict[str, list[VitalSample]] = Field(default_factory=dict)
    # per-signal quality governs (decision D)
    signal_quality: dict[str, float] = Field(default_factory=dict)
    pacer: Optional[dict] = None  # {info, offset} when present
    ground_truth: Optional[GroundTruth] = None  # sim files only

    model_config = {"arbitrary_types_allowed": True}  # Fraction


# --------------------------------------------------------------------------------------
# Inference + analysis plane
# --------------------------------------------------------------------------------------

TrendDirection = Literal["improving", "deteriorating", "stable", "insufficient_data"]
MEWSRisk = Literal["Low", "Medium", "High", "Critical"]


class MEWS(BaseModel):
    score: int
    risk: MEWSRisk


class VitalTrend(BaseModel):
    direction: TrendDirection
    p: Optional[float] = None  # Mann-Kendall p-value when available


class ClinicalAnalysis(BaseModel):
    """`VitalsAnalysis.analyze` output: MEWS + per-vital trend + ECG-vital correlations."""

    mews: MEWS
    vital_trends: dict[str, VitalTrend] = Field(default_factory=dict)
    care_guidance: list[str] = Field(default_factory=list)
    correlations: list[str] = Field(default_factory=list)


class DeviceEvent(BaseModel):
    """Inference + analysis output = `SignalWindow` + model prediction + clinical analysis."""

    window: SignalWindow
    event_type: str  # one of CLASS_NAMES, predicted by ECG_TransConv
    confidence: float
    is_false_positive: bool  # event_type == NORMAL_SINUS
    low_confidence: bool = False  # top-class confidence < LOW_CONFIDENCE_CAVEAT
    uncertain: bool = False  # NORMAL_SINUS below FP_SUPPRESS_MIN_CONFIDENCE
    analysis: ClinicalAnalysis
    report_md: str = ""  # per-event markdown report (D18)

    def model_post_init(self, __context) -> None:  # noqa: D401
        if self.event_type not in CLASS_NAMES:
            raise ValueError(f"unknown event_type: {self.event_type!r}")


# --------------------------------------------------------------------------------------
# Knowledge-base plane
# --------------------------------------------------------------------------------------


class Passage(BaseModel):
    """A retrieved vector chunk with its citation."""

    text: str
    source: str
    score: float


class Relationship(BaseModel):
    """A graph fact with its citation."""

    fact: str
    source: str
    score: float = 1.0


class RetrievalResult(BaseModel):
    """Two labelled blocks (D8); `relationships` is empty under `vector` mode."""

    query: str
    passages: list[Passage] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    mode: Literal["vector", "hybrid"] = "hybrid"


# --------------------------------------------------------------------------------------
# Conversation state
# --------------------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: Literal["clinician", "assistant", "system"]
    text: str
    timestamp: float


class ConversationState(BaseModel):
    """Persisted/resumed by the orchestrator (Redis checkpointer, Phase 3/4)."""

    session_id: str
    patient_ref: Optional[str] = None
    authenticated: bool = False
    turns: list[ChatTurn] = Field(default_factory=list)
