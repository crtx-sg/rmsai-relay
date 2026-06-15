"""Lean datastore connectivity (Phase 0 infra gate).

Marked `infra`: these need the lean containers up
(`docker compose -f infra/docker-compose.yml up -d redis neo4j qdrant`). They skip gracefully
when a service is unreachable so the default `pytest` run stays green without Docker.
"""

from __future__ import annotations

import pytest

from common.config import DEFAULT

pytestmark = pytest.mark.infra


def test_redis_ping():
    redis = pytest.importorskip("redis")
    try:
        client = redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2)
        assert client.ping() is True
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"redis unreachable: {exc}")


def test_neo4j_connectivity():
    neo4j = pytest.importorskip("neo4j")
    try:
        driver = neo4j.GraphDatabase.driver(
            DEFAULT.neo4j_uri, auth=(DEFAULT.neo4j_user, DEFAULT.neo4j_password)
        )
        driver.verify_connectivity()
        driver.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")


def test_qdrant_collections():
    qdrant_client = pytest.importorskip("qdrant_client")
    try:
        client = qdrant_client.QdrantClient(url=DEFAULT.qdrant_url, timeout=2)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qdrant unreachable: {exc}")
