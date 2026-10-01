"""Pytest session config.

Make the test suite hermetic: do NOT load the developer's `.env` (which may select real
backends like Presidio/Ollama that aren't installed in every environment). Tests pin the
backends they need explicitly; everything else uses the offline defaults.
"""

import os

os.environ.setdefault("RMSAI_NO_DOTENV", "1")

import pytest  # noqa: E402

# Module fixtures that connect to the LIVE Neo4j when it is reachable and `reset_all()` it. Any test
# using one is `infra`, so `pytest -m "not infra"` (the "offline" run) can never wipe a running
# demo graph. Keyed by module + fixture because the fixture names are generic.
_LIVE_GRAPH_FIXTURES = {
    "test_graph_templates.py": "graph",
    "test_hybrid.py": "retriever",
    "test_voice_orchestrator.py": "handler",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        fixture = _LIVE_GRAPH_FIXTURES.get(item.path.name)
        if fixture and fixture in item.fixturenames:
            item.add_marker(pytest.mark.infra)
