"""Deterministic evaluation cohort.

Seeds a fixed graph whose co-morbidity edges (graph-only facts, stated in no document) are the
discriminators between hybrid and vector-only:
  * atrial fibrillation + hypertension co-occur in 2 patients
  * heart failure + chronic kidney disease co-occur in 2 patients
"""

from __future__ import annotations

from pathlib import Path

EVAL_PATIENTS = [
    {
        "patient_id": "PE001", "gender": "M", "age": 76,
        "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
        "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"],
    },
    {
        "patient_id": "PE002", "gender": "F", "age": 71,
        "comorbidities": ["atrial fibrillation", "hypertension", "diabetes"],
        "prior_diagnoses": ["none"], "symptoms": ["dyspnea"], "surgeries": ["none"],
        "current_medications": ["ace-inhibitor"],
    },
    {
        "patient_id": "PE003", "gender": "M", "age": 80,
        "comorbidities": ["heart failure", "chronic kidney disease"], "prior_diagnoses": ["none"],
        "symptoms": ["dyspnea"], "surgeries": ["none"], "current_medications": ["diuretic"],
    },
    {
        "patient_id": "PE004", "gender": "F", "age": 67,
        "comorbidities": ["heart failure", "chronic kidney disease", "coronary artery disease"],
        "prior_diagnoses": ["none"], "symptoms": ["fatigue"], "surgeries": ["none"],
        "current_medications": ["statin"],
    },
]

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_PROTOCOLS = Path(__file__).resolve().parents[2] / "common" / "protocols" / "care_protocols.yaml"


def seed_eval_graph(driver) -> None:
    """Reset + populate the graph with the eval cohort, document entities, and protocols."""
    from kb.graph.extract import extract_dir
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record
    from kb.graph.protocols import load_protocol_file
    from kb.graph.schema import migrate

    driver.reset_all()
    migrate(driver)
    for patient in EVAL_PATIENTS:
        ingest_patient_record(driver, patient)
    derive_comorbidity(driver)
    extract_dir(driver, _DOCS)
    load_protocol_file(driver, _PROTOCOLS)
