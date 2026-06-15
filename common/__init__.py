"""Frozen contracts + cross-cutting utilities. Importing this module must not pull torch."""

from __future__ import annotations

from .audit import AuditLog
from .bed_assignment import BEDS_PER_UNIT, BedAssignmentStub
from .config import DEFAULT, Config
from .criticality import Criticality, at_least, criticality
from .ecg_model_stub import StubECGModel
from .event_types import (
    CLASS_NAMES,
    NORMAL_SINUS,
    EventType,
    is_false_positive,
    is_valid_event_type,
)
from .interfaces import ECGModel, EventStore, LLMProvider, VitalsAnalysis
from .patient_history import PatientHistory, PatientHistoryStub
from .protocol_loader import CareProtocol, ProtocolStep, load_protocols
from .redacting_logger import RedactingFilter, get_redacting_logger
from .schemas import (
    MEWS,
    ChatTurn,
    ClinicalAnalysis,
    ConversationState,
    DeviceEvent,
    Passage,
    Relationship,
    RetrievalResult,
    SignalWindow,
    Vital,
    WindowGeometry,
)

__all__ = [
    "AuditLog",
    "BEDS_PER_UNIT",
    "BedAssignmentStub",
    "Config",
    "DEFAULT",
    "Criticality",
    "at_least",
    "criticality",
    "StubECGModel",
    "CLASS_NAMES",
    "NORMAL_SINUS",
    "EventType",
    "is_false_positive",
    "is_valid_event_type",
    "ECGModel",
    "EventStore",
    "LLMProvider",
    "VitalsAnalysis",
    "PatientHistory",
    "PatientHistoryStub",
    "CareProtocol",
    "ProtocolStep",
    "load_protocols",
    "RedactingFilter",
    "get_redacting_logger",
    "ChatTurn",
    "ClinicalAnalysis",
    "ConversationState",
    "DeviceEvent",
    "MEWS",
    "Passage",
    "Relationship",
    "RetrievalResult",
    "SignalWindow",
    "Vital",
    "WindowGeometry",
]
