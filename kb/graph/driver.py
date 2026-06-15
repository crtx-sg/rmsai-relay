"""Neo4j connection wrapper + read-only guard.

`GraphDriver` is a thin helper over the official driver with separate read/write entry points and
a `assert_read_only` check used to keep the LLM text-to-Cypher fallback (Phase 2B.4) safe.
"""

from __future__ import annotations

import re

from neo4j import GraphDatabase

from common.config import DEFAULT, Config

# Clauses that mutate the graph — any match means the statement is NOT read-only.
_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH|DROP|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)
_APOC_WRITE = re.compile(r"apoc\.\w*(create|merge|delete|set|refactor)", re.IGNORECASE)


class NotReadOnlyError(Exception):
    """Raised when a statement expected to be read-only contains a write clause."""


def is_read_only(cypher: str) -> bool:
    return not (_WRITE_CLAUSE.search(cypher) or _APOC_WRITE.search(cypher))


def assert_read_only(cypher: str) -> None:
    if not is_read_only(cypher):
        raise NotReadOnlyError(f"refusing non-read-only Cypher: {cypher!r}")


class GraphDriver:
    def __init__(self, driver) -> None:
        self._driver = driver

    @classmethod
    def from_config(cls, config: Config = DEFAULT) -> "GraphDriver":
        return cls(
            GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run_write(self, cypher: str, **params) -> list[dict]:
        with self._driver.session() as session:
            return [r.data() for r in session.run(cypher, **params)]

    def run_read(self, cypher: str, **params) -> list[dict]:
        with self._driver.session() as session:
            return [r.data() for r in session.execute_read(lambda tx: list(tx.run(cypher, **params)))]

    def run_read_only(self, cypher: str, **params) -> list[dict]:
        """Run a statement only after asserting it is read-only (text-to-Cypher fallback)."""
        assert_read_only(cypher)
        return self.run_read(cypher, **params)

    def reset_all(self) -> None:
        """Delete every node + relationship (test helper)."""
        self.run_write("MATCH (n) DETACH DELETE n")
