"""Phase 4 event flow: inbound DeviceEvent -> MonitoredEvent + archived EventReport."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT
from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_DOCS = Path(__file__).resolve().parents[1] / "docs"

pytestmark = pytest.mark.infra


class _ForcedModel(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.91


def _device_event():
    w = next(read_hdf5_file(_FIXTURE))  # patient PT1155
    w.vitals["HR"].value = 145.0  # trigger an AFib-RVR care-guidance note
    return process_window(w, _ForcedModel(), MewsVitalsAnalysis())


@pytest.fixture()
def setup():
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
    ingest_patient_record(d, {
        "patient_id": "PT1155", "gender": "M", "age": 73,
        "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
        "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"],
    }, bed=("Unit1", "Unit1-Bed05"))

    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    vector.index_dir(_DOCS)
    yield d, vector
    d.reset_all()
    d.close()


def test_event_persisted_and_queryable_by_bed_and_time(setup):
    from kb.graph.templates import run_template
    from orchestrator.event_flow import process_device_event

    driver, vector = setup
    result = process_device_event(
        _device_event(), driver, vector, bed=("Unit1", "Unit1-Bed05"), generated_at=1000.0
    )

    # queryable by time (T2: positive events)
    positive = run_template(driver, "positive_events_since", since=0.0)
    assert any(r["event"] == "ATRIAL_FIBRILLATION" for r in positive)

    # queryable by bed (T3: status on bed)
    status = run_template(driver, "event_status_on_bed", bed="Unit1-Bed05")
    assert status and status[0]["reported_event"] == "ATRIAL_FIBRILLATION"

    # vitals snapshot stored inline (T5)
    vitals = run_template(driver, "vitals_at_event", event_uuid=result.event_uuid)
    assert vitals[0]["hr"] == 145.0


def test_action_items_from_care_guidance(setup):
    from kb.graph.templates import run_template
    from orchestrator.event_flow import process_device_event

    driver, vector = setup
    result = process_device_event(_device_event(), driver, vector, bed=("Unit1", "Unit1-Bed05"))
    assert result.action_items >= 1
    outstanding = run_template(driver, "outstanding_action_items")
    assert any("rate control" in r["action"].lower() or "afib" in r["action"].lower()
               for r in outstanding)


def test_report_archived_and_retrievable(setup):
    from kb.graph.templates import run_template
    from orchestrator.event_flow import process_device_event

    driver, vector = setup
    result = process_device_event(_device_event(), driver, vector, bed=("Unit1", "Unit1-Bed05"))

    # report node linked to the event (T4) and marked indexed
    reports = run_template(driver, "reports_for_bed", bed="Unit1-Bed05")
    assert reports and reports[0]["report_id"] == result.report_id
    idx = driver.run_read(
        "MATCH (r:Report {id:$rid}) RETURN r.index_status AS s", rid=result.report_id
    )
    assert idx[0]["s"] == "indexed"

    # report narrative retrievable from the vector store
    passages = vector.retrieve("atrial fibrillation event report patient context", k=5).passages
    assert any(p.source.startswith("report:") for p in passages)


def test_report_grounded_in_patient_context(setup):
    from orchestrator.event_flow import process_device_event

    driver, vector = setup
    result = process_device_event(_device_event(), driver, vector, bed=("Unit1", "Unit1-Bed05"))
    assert "Patient context" in result.report_md
    assert "atrial fibrillation" in result.report_md.lower()  # from graph history
    assert "palpitations" in result.report_md.lower()
