"""Patient-record ingestion + co-morbidity derivation (live neo4j)."""

from __future__ import annotations

import pytest

from common.config import DEFAULT

pytestmark = pytest.mark.infra


@pytest.fixture()
def driver():
    from kb.graph.driver import GraphDriver
    from kb.graph.schema import migrate

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    d.reset_all()
    migrate(d)
    yield d
    d.reset_all()
    d.close()


P1 = {
    "patient_id": "PT9001", "gender": "M", "age": 72,
    "comorbidities": ["hypertension", "atrial fibrillation"],
    "prior_diagnoses": ["none"], "symptoms": ["palpitations"],
    "surgeries": ["pacemaker"], "current_medications": ["beta-blocker", "anticoagulant"],
}
P2 = {
    "patient_id": "PT9002", "gender": "F", "age": 65,
    "comorbidities": ["hypertension", "atrial fibrillation", "diabetes"],
    "prior_diagnoses": ["none"], "symptoms": ["dyspnea"],
    "surgeries": ["none"], "current_medications": ["ace-inhibitor"],
}


def _count(driver, cypher) -> int:
    return driver.run_read(cypher)[0]["n"]


def test_ingest_node_and_edge_counts(driver):
    from kb.graph.ingest import ingest_patient_record

    ingest_patient_record(driver, P1, bed=("Unit1", "Unit1-Bed01"))
    ingest_patient_record(driver, P2, bed=("Unit1", "Unit1-Bed02"))

    assert _count(driver, "MATCH (p:Patient) RETURN count(p) AS n") == 2
    assert _count(driver, "MATCH (c:Condition) RETURN count(c) AS n") == 3  # htn, afib, diabetes
    assert _count(driver, "MATCH (:Patient)-[r:HAS_DIAGNOSIS]->() RETURN count(r) AS n") == 5
    assert _count(driver, "MATCH (s:Symptom) RETURN count(s) AS n") == 2
    assert _count(driver, "MATCH (su:Surgery) RETURN count(su) AS n") == 1  # 'none' skipped
    assert _count(driver, "MATCH (u:Unit) RETURN count(u) AS n") == 1
    assert _count(driver, "MATCH (b:Bed) RETURN count(b) AS n") == 2


def test_reingest_is_noop(driver):
    from kb.graph.ingest import ingest_patient_record

    ingest_patient_record(driver, P1, bed=("Unit1", "Unit1-Bed01"))
    before = _count(driver, "MATCH (n) RETURN count(n) AS n")
    rels_before = _count(driver, "MATCH ()-[r]->() RETURN count(r) AS n")
    ingest_patient_record(driver, P1, bed=("Unit1", "Unit1-Bed01"))  # re-ingest
    assert _count(driver, "MATCH (n) RETURN count(n) AS n") == before
    assert _count(driver, "MATCH ()-[r]->() RETURN count(r) AS n") == rels_before


def test_shared_condition_node_is_single(driver):
    from kb.graph.ingest import ingest_patient_record

    ingest_patient_record(driver, P1)
    ingest_patient_record(driver, P2)
    # both patients' 'atrial fibrillation' resolve to ONE node
    assert _count(
        driver, "MATCH (c:Condition {id:'atrial_fibrillation'}) RETURN count(c) AS n"
    ) == 1
    assert _count(
        driver,
        "MATCH (:Patient)-[:HAS_DIAGNOSIS]->(c:Condition {id:'atrial_fibrillation'}) "
        "RETURN count(*) AS n",
    ) == 2


def test_comorbidity_rule_fires_only_on_evidence(driver):
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record

    ingest_patient_record(driver, P1)
    ingest_patient_record(driver, P2)
    edges = derive_comorbidity(driver, min_co_occurrence=2)
    # htn + afib co-occur in 2 patients -> exactly one edge; diabetes (1 patient) excluded.
    assert edges == 1
    row = driver.run_read(
        "MATCH (:Condition {id:'atrial_fibrillation'})-[r:CO_MORBID_WITH]-"
        "(:Condition {id:'hypertension'}) "
        "RETURN r.co_occurrence_count AS cooc, r.confidence AS conf, r.source AS src"
    )[0]
    assert row["cooc"] == 2
    assert row["conf"] == pytest.approx(1.0)
    assert row["src"] == "cohort-co-occurrence"


def test_comorbidity_rebuild_is_idempotent(driver):
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record

    ingest_patient_record(driver, P1)
    ingest_patient_record(driver, P2)
    assert derive_comorbidity(driver) == derive_comorbidity(driver) == 1


def test_manages_edges(driver):
    from kb.graph.ingest import ingest_patient_record

    ingest_patient_record(driver, P1)
    ingest_patient_record(driver, P2)
    # beta-blocker -> atrial fibrillation MANAGES edge exists
    assert _count(
        driver,
        "MATCH (:Treatment {id:'beta_blocker'})-[:MANAGES]->(:Condition {id:'atrial_fibrillation'})"
        " RETURN count(*) AS n",
    ) == 1
