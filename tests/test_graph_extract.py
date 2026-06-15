"""Document extraction + protocol loading: shared nodes, allowlist, idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT
from kb.graph.extract import (
    AllowlistError,
    assert_edge_allowed,
    assert_label_allowed,
)

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PROTOCOLS = Path(__file__).resolve().parents[1] / "common" / "protocols" / "care_protocols.yaml"


# --- allowlist (no DB) ---


def test_allowlist_accepts_known():
    assert_label_allowed("Condition")
    assert_edge_allowed("Guideline", "APPLIES_TO", "Condition")


def test_allowlist_rejects_unknown_label():
    with pytest.raises(AllowlistError):
        assert_label_allowed("Patient")  # extraction may not write Patient nodes


def test_allowlist_rejects_unknown_edge():
    with pytest.raises(AllowlistError):
        assert_edge_allowed("Guideline", "HAS_DIAGNOSIS", "Condition")


# --- extraction against live neo4j ---


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


def _count(driver, cypher) -> int:
    return driver.run_read(cypher)[0]["n"]


@pytest.mark.infra
def test_extract_creates_guidelines_and_edges(driver):
    from kb.graph.extract import extract_dir

    summary = extract_dir(driver, _DOCS)
    assert summary["chunks"] > 0
    assert _count(driver, "MATCH (g:Guideline) RETURN count(g) AS n") > 0
    # afib guideline APPLIES_TO atrial fibrillation, with a citation source
    rows = driver.run_read(
        "MATCH (g:Guideline)-[r:APPLIES_TO]->(c:Condition {id:'atrial_fibrillation'}) "
        "RETURN r.source AS src LIMIT 1"
    )
    assert rows and rows[0]["src"]


@pytest.mark.infra
def test_patient_and_protocol_share_one_condition_node(driver):
    from kb.graph.extract import extract_dir
    from kb.graph.ingest import ingest_patient_record

    # Patient with atrial fibrillation
    ingest_patient_record(driver, {
        "patient_id": "PT9100", "gender": "M", "age": 70,
        "comorbidities": ["atrial fibrillation"], "prior_diagnoses": ["none"],
        "symptoms": [], "surgeries": ["none"], "current_medications": [],
    })
    # Documents also mention atrial fibrillation
    extract_dir(driver, _DOCS)
    # Must be exactly ONE atrial_fibrillation node, reachable from BOTH a Patient and a Guideline.
    assert _count(driver, "MATCH (c:Condition {id:'atrial_fibrillation'}) RETURN count(c) AS n") == 1
    assert _count(
        driver,
        "MATCH (:Patient)-[:HAS_DIAGNOSIS]->(c:Condition {id:'atrial_fibrillation'})"
        "<-[:APPLIES_TO]-(:Guideline) RETURN count(DISTINCT c) AS n",
    ) == 1


@pytest.mark.infra
def test_reextract_is_noop(driver):
    from kb.graph.extract import extract_dir

    extract_dir(driver, _DOCS)
    n = _count(driver, "MATCH (n) RETURN count(n) AS n")
    r = _count(driver, "MATCH ()-[x]->() RETURN count(x) AS n")
    extract_dir(driver, _DOCS)  # re-extract
    assert _count(driver, "MATCH (n) RETURN count(n) AS n") == n
    assert _count(driver, "MATCH ()-[x]->() RETURN count(x) AS n") == r


@pytest.mark.infra
def test_load_protocols_links_to_shared_condition(driver):
    from kb.graph.ingest import ingest_patient_record
    from kb.graph.protocols import load_protocol_file

    ingest_patient_record(driver, {
        "patient_id": "PT9200", "gender": "F", "age": 60,
        "comorbidities": ["atrial fibrillation"], "prior_diagnoses": ["none"],
        "symptoms": [], "surgeries": ["none"], "current_medications": [],
    })
    n = load_protocol_file(driver, _PROTOCOLS)
    assert n == 2  # afib_rvr + default
    # afib protocol APPLIES_TO the SAME atrial_fibrillation node the patient is diagnosed with
    assert _count(
        driver,
        "MATCH (:Patient)-[:HAS_DIAGNOSIS]->(c:Condition {id:'atrial_fibrillation'})"
        "<-[:APPLIES_TO]-(:CareProtocol {id:'afib_rvr'}) RETURN count(*) AS n",
    ) == 1
    assert _count(
        driver, "MATCH (:CareProtocol {id:'afib_rvr'})-[:HAS_STEP]->(s) RETURN count(s) AS n"
    ) == 5
