"""Care-protocol loader skeleton (D19 / §5.5).

Parses curated care protocols from external YAML/JSON into `CareProtocol`/`ProtocolStep` objects
and renders each to citable narrative text. The graph + vector **writes** are wired in Phase 2B;
Phase 0 freezes the schema, the parser, and the matcher so the config format is stable.

A protocol is matched on `event_type` + vital conditions + minimum severity (most-specific-wins,
with a default fallback) — not event type alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

StepKind = Literal["test", "vitals_check", "medication", "escalation", "monitoring"]


class ProtocolMatch(BaseModel):
    event_type: str  # "*" = any
    vital_conditions: list[str] = Field(default_factory=list)  # e.g. ["HR > 120", "SBP < 90"]
    min_severity: str = "Low"


class ProtocolStep(BaseModel):
    order: int
    kind: StepKind
    text: str = ""
    # kind-specific fields kept loose at the schema level (validated per-kind later)
    fields: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in self.fields.items())
        body = self.text or self.kind
        return f"{self.order}. [{self.kind}] {body}" + (f" ({detail})" if detail else "")


class CareProtocol(BaseModel):
    id: str
    title: str
    version: str = "1"
    source: str = ""
    match: ProtocolMatch
    steps: list[ProtocolStep] = Field(default_factory=list)

    def render(self) -> str:
        """Render to narrative text for the vector store (Phase 2B indexes this)."""
        lines = [f"# {self.title}", f"Source: {self.source or 'TODO'}", ""]
        lines += [s.render() for s in sorted(self.steps, key=lambda s: s.order)]
        return "\n".join(lines)


_KNOWN_STEP_FIELDS = {"order", "kind", "text"}


def _parse_step(raw: dict[str, Any]) -> ProtocolStep:
    fields = {k: v for k, v in raw.items() if k not in _KNOWN_STEP_FIELDS}
    return ProtocolStep(
        order=int(raw["order"]),
        kind=raw["kind"],
        text=str(raw.get("text", "")),
        fields=fields,
    )


def _parse_protocol(raw: dict[str, Any]) -> CareProtocol:
    m = raw.get("match", {}) or {}
    return CareProtocol(
        id=raw["id"],
        title=raw["title"],
        version=str(raw.get("version", "1")),
        source=str(raw.get("source", "")),
        match=ProtocolMatch(
            event_type=str(m.get("event_type", "*")),
            vital_conditions=list(m.get("vital_conditions", []) or []),
            min_severity=str(m.get("min_severity", "Low")),
        ),
        steps=[_parse_step(s) for s in raw.get("steps", [])],
    )


def load_protocols(path: str | Path) -> list[CareProtocol]:
    """Load and parse a YAML/JSON protocol config file into `CareProtocol` objects."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, list):
        raise ValueError("protocol config must be a list of protocols")
    return [_parse_protocol(p) for p in data]


def find_default(protocols: list[CareProtocol]) -> Optional[CareProtocol]:
    """Return the fallback protocol (match.event_type == '*'), if any."""
    for p in protocols:
        if p.match.event_type == "*":
            return p
    return None
