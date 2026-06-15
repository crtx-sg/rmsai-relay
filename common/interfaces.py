"""Abstract interfaces — the swappable seams.

Each has a POC implementation (stub or vendored-wrapper) elsewhere in `common/` or in a
subsystem package; the interface exists so the real backend swaps in without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .schemas import ClinicalAnalysis, SignalWindow

# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------


class LLMProvider(ABC):
    """`LocalProvider` is the default; Anthropic/OpenAI are synthetic-data-only options."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------------------
# ECG model + vitals analysis
# --------------------------------------------------------------------------------------


class ECGModel(ABC):
    """Wraps the `ecgtranscnn` 7-lead classifier. Stub stands in until weights are present."""

    @abstractmethod
    def predict(self, window: SignalWindow) -> tuple[str, float]:
        """Return `(event_type, confidence)` — `event_type` is one of the 16 classes."""


class VitalsAnalysis(ABC):
    """Wraps `ecgtranscnn.mews` (MEWS + Mann-Kendall trend + ECG-vital correlation)."""

    @abstractmethod
    def analyze(self, window: SignalWindow) -> ClinicalAnalysis: ...


# --------------------------------------------------------------------------------------
# Operational event store (repository — Neo4j for POC, Postgres/Timescale later)
# --------------------------------------------------------------------------------------


class EventStore(ABC):
    """Persists/queries the operational event log (MonitoredEvent, ActionItem, Report)."""

    @abstractmethod
    def persist_event(self, event: dict) -> str:
        """Persist a MonitoredEvent (idempotent by uuid); return its id."""

    @abstractmethod
    def get_event(self, uuid: str) -> Optional[dict]: ...

    @abstractmethod
    def run_template(self, name: str, params: dict) -> list[dict]:
        """Run a named, parameterized, read-only Cypher template."""
