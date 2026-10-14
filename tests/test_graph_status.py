"""Event lifecycle in the graph: a replay or a late call result never reopens a closed event.

Needs a Neo4j it may wipe (RMSAI_ALLOW_LIVE_GRAPH_RESET=1); skipped otherwise, like the other
graph-backed tests.
"""

from __future__ import annotations

import pytest

from common.config import DEFAULT
from kb.graph.events import get_event_status, persist_monitored_event, reopen_events, set_event_status

pytestmark = pytest.mark.infra


@pytest.fixture()
def graph():
    from kb.graph.driver import GraphDriver
    from kb.graph.ingest import ingest_patient_record
    from kb.graph.schema import migrate

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    d.reset_all()
    migrate(d)
    ingest_patient_record(d, {"patient_id": "PT9001", "comorbidities": []}, bed=("Unit1", "Unit1-Bed01"))
    yield d
    d.reset_all()
    d.close()


def _persist(d, **kw):
    persist_monitored_event(
        d, uuid="E1", patient_id="PT9001", timestamp=1000.0, event_type="ATRIAL_FIBRILLATION",
        confidence=0.9, is_false_positive=False, status="reported",
        action_items=[{"text": "Rate control", "priority": "high"}], **kw)


def _action_status(d):
    return d.run_read("MATCH (:MonitoredEvent {id:'E1'})-[:HAS_ACTION]->(a) RETURN a.status AS s")[0]["s"]


def test_replay_keeps_acknowledgment(graph):
    _persist(graph)
    set_event_status(graph, "E1", "acknowledged")
    graph.run_write("MATCH (a:ActionItem) SET a.status='done'")
    _persist(graph)  # the same event re-ingested
    assert get_event_status(graph, "E1") == "acknowledged"
    assert _action_status(graph) == "done"


def test_open_status_never_replaces_closed(graph):
    _persist(graph)
    set_event_status(graph, "E1", "acknowledged")
    set_event_status(graph, "E1", "notify_failed")  # the queued call rang out afterwards
    assert get_event_status(graph, "E1") == "acknowledged"
    set_event_status(graph, "E1", "resolved")  # closed -> closed is allowed
    assert get_event_status(graph, "E1") == "resolved"


def test_reopen_resets_status_and_action_items(graph):
    _persist(graph)
    set_event_status(graph, "E1", "acknowledged")
    graph.run_write("MATCH (a:ActionItem) SET a.status='done'")
    assert reopen_events(graph) == 1
    assert get_event_status(graph, "E1") == "reported" and _action_status(graph) == "outstanding"


def test_reassigned_patient_has_one_current_bed(graph):
    from kb.graph.ingest import ASSIGN_BED

    graph.run_write(ASSIGN_BED, unit="Unit1", bed="Unit1-Bed02", pid="PT9001")
    rows = graph.run_read("MATCH (:Patient {id:'PT9001'})-[r:ASSIGNED_TO]->(b) "
                          "WHERE r.current RETURN b.label AS bed")
    assert [r["bed"] for r in rows] == ["Unit1-Bed02"]
