"""Pytest session config.

Make the test suite hermetic: do NOT load the developer's `.env` (which may select real
backends like Presidio/Ollama that aren't installed in every environment). Tests pin the
backends they need explicitly; everything else uses the offline defaults.
"""

import os

os.environ.setdefault("RMSAI_NO_DOTENV", "1")

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

# Every audit write in the suite goes to a throwaway file. Set at import, before any test module
# imports `common.config`, so `DEFAULT.audit_log_path` and bare `AuditLog()` both pick it up and a
# test can never append to the developer's real `data/audit.jsonl`.
os.environ["AUDIT_LOG_PATH"] = str(Path(tempfile.mkdtemp(prefix="rmsai-audit-")) / "audit.jsonl")

import pytest  # noqa: E402

_REAL_AUDIT_LOG = Path(__file__).resolve().parent / "data" / "audit.jsonl"


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


@pytest.fixture(autouse=True)
def _real_audit_log_untouched():
    """Fail the test that writes to the real audit log (a leak past AUDIT_LOG_PATH)."""
    before = _size(_REAL_AUDIT_LOG)
    yield
    after = _size(_REAL_AUDIT_LOG)
    assert after == before, (
        f"test wrote {after - before} bytes to {_REAL_AUDIT_LOG}; audit writes must go to "
        "AUDIT_LOG_PATH or an explicit tmp path")

# Module fixtures that connect to the LIVE Neo4j when it is reachable and `reset_all()` it. Any test
# using one is `infra`, so `pytest -m "not infra"` (the "offline" run) can never wipe a running
# demo graph. Keyed by module + fixture because the fixture names are generic.
_LIVE_GRAPH_FIXTURES = {
    "test_graph_templates.py": "graph",
    "test_hybrid.py": "retriever",
    "test_voice_orchestrator.py": "handler",
}


# The `infra` marker alone did not protect the demo graph: a run without `-m "not infra"` (e.g.
# `pytest tests -k call`, which selects test_authenticated_caller_gets_grounded_answer) wiped every
# patient, bed and event. These tests now also require an explicit opt-in.
_ALLOW_LIVE_GRAPH_RESET = "RMSAI_ALLOW_LIVE_GRAPH_RESET"


def pytest_collection_modifyitems(config, items):
    allow_reset = os.environ.get(_ALLOW_LIVE_GRAPH_RESET) == "1"
    for item in items:
        fixture = _LIVE_GRAPH_FIXTURES.get(item.path.name)
        if fixture and fixture in item.fixturenames:
            item.add_marker(pytest.mark.infra)
            if not allow_reset:
                item.add_marker(pytest.mark.skip(
                    reason=f"wipes the LIVE Neo4j graph (reset_all); set {_ALLOW_LIVE_GRAPH_RESET}=1 "
                           "to run it against a graph you can lose"))
