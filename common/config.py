"""POC configuration defaults (all overridable via environment / .env).

Every value here is a deliberate POC simplification to revisit for production (spec §13).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # Confidence thresholds (G7)
    fp_suppress_min_confidence: float = 0.80  # suppress as FP only above this; below ⇒ uncertain
    low_confidence_caveat: float = 0.60  # top-class < this ⇒ low_confidence + caveat

    # Inbound auth (G10)
    inbound_auth_pin: str = "1234"

    # Outbound calling (§6.1 / D16)
    outbound_enabled: bool = False
    outbound_call_number: str = ""
    outbound_from: str = ""
    outbound_min_criticality: str = "High"
    outbound_max_retries: int = 2
    outbound_retry_delay_s: int = 30

    # Datastore endpoints (lean services)
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rmsai_dev_pw"
    qdrant_url: str = "http://localhost:6333"

    # Audit log
    audit_log_path: str = "data/audit.jsonl"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            fp_suppress_min_confidence=_f("FP_SUPPRESS_MIN_CONFIDENCE", 0.80),
            low_confidence_caveat=_f("LOW_CONFIDENCE_CAVEAT", 0.60),
            inbound_auth_pin=os.environ.get("INBOUND_AUTH_PIN", "1234"),
            outbound_enabled=_b("OUTBOUND_ENABLED", False),
            outbound_call_number=os.environ.get("OUTBOUND_CALL_NUMBER", ""),
            outbound_from=os.environ.get("OUTBOUND_FROM", ""),
            outbound_min_criticality=os.environ.get("OUTBOUND_MIN_CRITICALITY", "High"),
            outbound_max_retries=_i("OUTBOUND_MAX_RETRIES", 2),
            outbound_retry_delay_s=_i("OUTBOUND_RETRY_DELAY_S", 30),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "rmsai_dev_pw"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            audit_log_path=os.environ.get("AUDIT_LOG_PATH", "data/audit.jsonl"),
        )


#: Process-wide default; call `Config.from_env()` for a fresh read.
DEFAULT = Config()
