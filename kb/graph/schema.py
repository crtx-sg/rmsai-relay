"""Graph schema migration — idempotent uniqueness constraints + indexes.

Every node label gets a uniqueness constraint on `id` (which also creates the backing index).
A couple of extra indexes support the time-range / criticality operational queries (T1/T2).
"""

from __future__ import annotations

from .driver import GraphDriver

# Every node label in the graph schema (kickoff §Graph schema).
NODE_LABELS = (
    "Patient",
    "Condition",
    "Treatment",
    "Symptom",
    "Surgery",
    "Guideline",
    "Unit",
    "Bed",
    "MonitoredEvent",
    "ActionItem",
    "Report",
    "CareProtocol",
    "ProtocolStep",
)

# Extra non-unique indexes for the operational access patterns.
_EXTRA_INDEXES = (
    ("MonitoredEvent", "timestamp"),
    ("MonitoredEvent", "criticality"),
    ("MonitoredEvent", "uuid"),
    ("Bed", "label"),
)


def migrate(driver: GraphDriver) -> None:
    """Create all constraints + indexes (idempotent — safe to run repeatedly)."""
    for label in NODE_LABELS:
        driver.run_write(
            f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
    for label, prop in _EXTRA_INDEXES:
        driver.run_write(
            f"CREATE INDEX {label.lower()}_{prop}_idx IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop})"
        )


def list_constraints(driver: GraphDriver) -> list[dict]:
    return driver.run_read("SHOW CONSTRAINTS")
