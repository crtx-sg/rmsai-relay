"""Phase 7 full-loop CLI smoke test: drop HDF5 -> detect -> call -> follow-up -> ack."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))

pytestmark = pytest.mark.infra


def test_full_loop_cli(capsys):
    from cli.outbound import main
    from kb.graph.driver import GraphDriver

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
        import redis
        redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2).ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend unreachable: {exc}")
    d.reset_all()
    d.close()

    rc = main([
        "--file", str(_FIXTURE),
        "--follow-up", "what is the rate control for atrial fibrillation",
        "--ack", "yes I acknowledge",
        "--embedder", "hashing",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # the committed fixture + deterministic stub yields an SVT (High) event -> a call + ack
    assert "[CALL]" in out
    assert "status: acknowledged" in out
