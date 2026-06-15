"""Lightweight tracing — spans across orchestrator/flow steps.

A minimal, dependency-free tracer (the POC stand-in for LangGraph/OTel tracing): each pipeline
step opens a `span` context manager that records its name, attributes, and duration. Spans are
collected on the `Tracer` and surfaced (e.g. on `TurnResult.trace`) so failure-mode behaviour and
latency are observable. Swap for OpenTelemetry without changing call sites.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "duration_ms": round(self.duration_ms, 3), **self.attributes}
        if self.error:
            d["error"] = self.error
        return d


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes):
        sp = Span(name=name, attributes=dict(attributes))
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as exc:  # record then re-raise
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.spans.append(sp)

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]
