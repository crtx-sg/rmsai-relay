"""Operational templates (T1-T10 + relationship lookups) on a fixed synthetic graph + lookup guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import DEFAULT
from kb.graph.driver import NotReadOnlyError

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PROTOCOLS = Path(__file__).resolve().parents[1] / "common" / "protocols" / "care_protocols.yaml"

NOW = 1_000_000.0  # fixed clock for deterministic time windows

P1 = {
    "patient_id": "PT8001", "gender": "M", "age": 75,
    "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
    "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"],
}
P2 = {
    "patient_id": "PT8002", "gender": "F", "age": 68,
    "comorbidities": ["atrial fibrillation", "hypertension", "diabetes"],
    "prior_diagnoses": ["none"], "symptoms": ["dyspnea"], "surgeries": ["none"],
    "current_medications": ["ace-inhibitor"],
}


@pytest.fixture(scope="module")
def graph():
    from kb.graph.driver import GraphDriver
    from kb.graph.events import persist_monitored_event
    from kb.graph.extract import extract_dir
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record
    from kb.graph.protocols import load_protocol_file
    from kb.graph.schema import migrate

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unreachable: {exc}")
    d.reset_all()
    migrate(d)

    ingest_patient_record(d, P1, bed=("Unit1", "Unit1-Bed01"))
    ingest_patient_record(d, P2, bed=("Unit1", "Unit1-Bed02"))
    derive_comorbidity(d)
    extract_dir(d, _DOCS)
    load_protocol_file(d, _PROTOCOLS)

    vit = {"hr": 170, "sbp": 100, "dbp": 70, "spo2": 92, "rr": 22, "temp": 98.6}
    # e1: critical VF on bed01 (oldest)
    persist_monitored_event(
        d, uuid="evt-vf", patient_id="PT8001", timestamp=NOW - 3600,
        event_type="VENTRICULAR_FIBRILLATION", confidence=0.97, is_false_positive=False,
        mews_risk="High", ground_truth_condition="ventricular fibrillation", status="reported",
        vitals=vit, bed=("Unit1", "Unit1-Bed01"),
        action_items=[{"text": "Initiate ACLS protocol", "priority": "high"}],
        signal_ref="hdf5://PT8001/evt-vf", ecg_plot_ref="plots/evt-vf.png",
    )
    # e2: AFib on bed01 (newest on the bed)
    persist_monitored_event(
        d, uuid="evt-afib", patient_id="PT8001", timestamp=NOW - 1800,
        event_type="ATRIAL_FIBRILLATION", confidence=0.88, is_false_positive=False,
        mews_risk="Medium", ground_truth_condition="atrial fibrillation", status="reported",
        vitals=vit, bed=("Unit1", "Unit1-Bed01"),
        signal_ref="hdf5://PT8001/evt-afib", ecg_plot_ref="plots/evt-afib.png",
        vitals_plot_ref="plots/evt-afib-vitals.png",
    )
    # e3: false-positive NORMAL_SINUS on bed02
    persist_monitored_event(
        d, uuid="evt-fp", patient_id="PT8002", timestamp=NOW - 600,
        event_type="NORMAL_SINUS", confidence=0.95, is_false_positive=True,
        mews_risk="Low", status="reported", vitals=vit, bed=("Unit1", "Unit1-Bed02"),
    )
    yield d
    d.reset_all()
    d.close()


def _tpl(graph, name, **params):
    from kb.graph.templates import run_template

    return run_template(graph, name, **params)


def test_t1_critical_events(graph):
    rows = _tpl(graph, "critical_events_since", since=NOW - 24 * 3600)
    assert len(rows) == 1
    assert rows[0]["event"] == "VENTRICULAR_FIBRILLATION"
    assert rows[0]["bed"] == "Unit1-Bed01" and rows[0]["unit"] == "Unit1"


def test_t2_positive_events(graph):
    rows = _tpl(graph, "positive_events_since", since=NOW - 2 * 3600)
    events = {r["event"] for r in rows}
    assert "NORMAL_SINUS" not in events  # FP excluded
    assert {"VENTRICULAR_FIBRILLATION", "ATRIAL_FIBRILLATION"} <= events


def test_t3_event_status_on_bed(graph):
    rows = _tpl(graph, "event_status_on_bed", bed="Unit1-Bed01")
    assert len(rows) == 2
    assert rows[0]["ts"] >= rows[1]["ts"]  # newest first
    assert rows[0]["actual_condition"] == "atrial fibrillation"


def test_t5_vitals_at_event(graph):
    rows = _tpl(graph, "vitals_at_event", event_uuid="evt-vf")
    assert rows[0]["hr"] == 170 and rows[0]["spo2"] == 92


def test_vitals_at_patient_last_event(graph):
    # PT8001 has evt-vf (older) and evt-afib (newer); "last event" must resolve to evt-afib.
    rows = _tpl(graph, "vitals_at_patient_last_event", patient_id="PT8001")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "ATRIAL_FIBRILLATION"
    assert rows[0]["hr"] == 170 and rows[0]["spo2"] == 92


def test_vitals_at_patient_last_event_unknown_patient(graph):
    # Resilience: no event for the patient -> empty rows, not an error.
    assert _tpl(graph, "vitals_at_patient_last_event", patient_id="PT_NOPE") == []


def test_t6_outstanding_action_items(graph):
    rows = _tpl(graph, "outstanding_action_items")
    assert any("ACLS" in r["action"] for r in rows)


def test_t7_protocol_for_bed_last_event(graph):
    rows = _tpl(graph, "protocol_for_bed_last_event", bed="Unit1-Bed01")
    # last event on the bed is AFib -> afib_rvr protocol with its steps
    assert rows[0]["condition"] == "atrial fibrillation"
    assert rows[0]["protocol"] == "Atrial fibrillation with rapid ventricular response"
    assert len(rows[0]["steps"]) == 5


def test_t8_cohort_patterns(graph):
    rows = _tpl(graph, "cohort_patterns")
    assert rows
    assert all("event" in r and "age_band" in r for r in rows)


def test_t9_ecg_strips_last_event(graph):
    rows = _tpl(graph, "ecg_strips_last_event", patient="PT8001", event_type="ATRIAL_FIBRILLATION")
    assert rows[0]["signal_ref"] == "hdf5://PT8001/evt-afib"
    assert rows[0]["ecg_plot"] == "plots/evt-afib.png"


def test_t10_trend_last_event(graph):
    rows = _tpl(graph, "trend_last_event", patient="PT8001", event_type="ATRIAL_FIBRILLATION")
    assert rows[0]["vitals_plot"] == "plots/evt-afib-vitals.png"
    assert rows[0]["hr"] == 170


def test_comorbidity_neighborhood(graph):
    rows = _tpl(graph, "comorbidity_neighborhood", condition_id="atrial_fibrillation")
    names = {r["comorbidity"] for r in rows}
    assert "hypertension" in names  # afib + htn co-occur in 2 patients
    assert all(r["co_occurrence"] >= 2 for r in rows)


def test_guidance_for_condition(graph):
    rows = _tpl(graph, "guidance_for_condition", condition_id="atrial_fibrillation")
    row = rows[0]
    assert "Atrial fibrillation with rapid ventricular response" in row["protocols"]
    assert row["guidelines"]  # at least one extracted guideline applies


# --- lookup: intent match + read-only guard ---


def test_lookup_intent_critical_events(graph):
    from kb.graph.lookup import lookup

    result = lookup(graph, "show critical events in the last 24 hours", now=NOW)
    assert result["mode"] == "template"
    assert result["template"] == "critical_events_since"
    assert len(result["rows"]) == 1


def test_lookup_rejects_write_cypher(graph):
    from kb.graph.lookup import lookup

    with pytest.raises(NotReadOnlyError):
        lookup(graph, "MATCH (n) DETACH DELETE n", now=NOW)
