"""FOLLOWED_BY: each patient's events chained in time order, rebuilt on every persist."""

from __future__ import annotations

import uuid as _uuid

import pytest

from kb.graph.events import link_patient_events, persist_monitored_event


class _RecordingDriver:
    def __init__(self):
        self.writes: list[tuple[str, dict]] = []

    def run_write(self, cypher, **params):
        self.writes.append((cypher, params))


def test_persist_rebuilds_the_patients_chain():
    d = _RecordingDriver()
    persist_monitored_event(d, uuid="e1", patient_id="PT1", timestamp=100.0,
                            event_type="VENTRICULAR_TACHYCARDIA", confidence=0.9,
                            is_false_positive=False)
    chain = [(c, p) for c, p in d.writes if "FOLLOWED_BY" in c]
    assert [p["pid"] for _, p in chain] == ["PT1", "PT1"]
    assert "DELETE f" in chain[0][0]                                  # the old chain goes first…
    assert "ORDER BY e.timestamp, e.id" in chain[1][0]                # …then rebuilt in time order
    assert "gap_s" in chain[1][0]


def test_chain_is_scoped_to_one_patient():
    d = _RecordingDriver()
    link_patient_events(d, "PT2")
    assert all(p == {"pid": "PT2"} for _, p in d.writes)
    assert all("Patient {id:$pid}" in c for c, _ in d.writes)


# --- against Neo4j: a throwaway patient, removed afterwards (never reset_all) ---------------------

@pytest.fixture()
def neo4j():
    from common.config import DEFAULT
    from kb.graph.driver import GraphDriver

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    pid = f"PTCHAIN{_uuid.uuid4().hex[:8]}"
    d.run_write("MERGE (:Patient {id:$pid})", pid=pid)
    yield d, pid
    d.run_write("MATCH (p:Patient {id:$pid}) OPTIONAL MATCH (p)-[:HAD_EVENT]->(e) "
                "DETACH DELETE p, e", pid=pid)
    d.close()


@pytest.mark.infra
def test_out_of_order_events_chain_in_time_order(neo4j):
    d, pid = neo4j
    ids = {}
    for name, ts in (("late", 300.0), ("early", 100.0), ("mid", 200.0)):  # arrive out of order
        ids[name] = f"{pid}-{name}"
        persist_monitored_event(d, uuid=ids[name], patient_id=pid, timestamp=ts,
                                event_type="PVC", confidence=0.8, is_false_positive=False)
    rows = d.run_read(
        "MATCH (:Patient {id:$pid})-[:HAD_EVENT]->(a)-[f:FOLLOWED_BY]->(b) "
        "RETURN a.id AS a, b.id AS b, f.gap_s AS gap, a.timestamp AS t ORDER BY t", pid=pid)
    assert [(r["a"], r["b"], r["gap"]) for r in rows] == [
        (ids["early"], ids["mid"], 100.0), (ids["mid"], ids["late"], 100.0)]
    persist_monitored_event(d, uuid=ids["mid"], patient_id=pid, timestamp=200.0,  # a replay
                            event_type="PVC", confidence=0.8, is_false_positive=False)
    assert d.run_read("MATCH (:Patient {id:$pid})-[:HAD_EVENT]->()-[f:FOLLOWED_BY]->() "
                      "RETURN count(f) AS n", pid=pid)[0]["n"] == 2
