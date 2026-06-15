"""Neo4j driver read-only guard (unit) + schema migration (infra)."""

from __future__ import annotations

import pytest

from common.config import DEFAULT
from kb.graph.driver import NotReadOnlyError, assert_read_only, is_read_only


# --- read-only guard (no DB needed) ---


def test_read_only_accepts_match():
    assert is_read_only("MATCH (n:Patient) RETURN n LIMIT 5")


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DETACH DELETE n",
        "CREATE (n:Patient {id:'x'})",
        "MATCH (n) SET n.x = 1",
        "MERGE (n:Condition {id:'c'})",
        "MATCH (n) REMOVE n.x",
        "CALL apoc.create.node(['X'], {})",
    ],
)
def test_read_only_rejects_writes(cypher):
    assert not is_read_only(cypher)
    with pytest.raises(NotReadOnlyError):
        assert_read_only(cypher)


# --- migration (live neo4j) ---


def _driver_or_skip():
    from kb.graph.driver import GraphDriver

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
        return d
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")


@pytest.mark.infra
def test_migration_is_idempotent():
    from kb.graph.schema import NODE_LABELS, list_constraints, migrate

    driver = _driver_or_skip()
    try:
        migrate(driver)
        migrate(driver)  # second run must not error
        names = {c.get("name", "") for c in list_constraints(driver)}
        # every label has its uniqueness constraint
        for label in NODE_LABELS:
            assert f"{label.lower()}_id_unique" in names
    finally:
        driver.close()
