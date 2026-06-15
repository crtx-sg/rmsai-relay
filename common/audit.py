"""Append-only JSONL audit log (G14).

Captures `{ts, actor, action, subject, outcome, **extra}` for security-relevant events: PHI
reads, outbound calls + outcome, inbound auth results, queries run, and (Phase 9) live/camera
sessions. `subject` is always a pseudonym — never a name. Durable store + retention is Phase 8;
this is the POC file stub.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-naive append-only writer. One JSON object per line."""

    def __init__(self, path: str | os.PathLike = "data/audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        actor: str,
        action: str,
        subject: str,
        outcome: str,
        ts: float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append one audit record and return it. `subject` must be a pseudonym."""
        record: dict[str, Any] = {
            "ts": ts if ts is not None else time.time(),
            "actor": actor,
            "action": action,
            "subject": subject,
            "outcome": outcome,
        }
        if extra:
            record["extra"] = extra
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read back every record (test/debug convenience)."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
