"""EMR / FHIR client (stub path; HAPI path is integration-only)."""

from __future__ import annotations

from emr.fhir import StubFhirClient, get_fhir_client


def test_stub_returns_history_shape():
    h = StubFhirClient().get_patient_history("PT4242")
    assert h["patient_id"] == "PT4242"
    assert h["gender"] in {"M", "F"}
    assert 18 <= h["age"] <= 95
    for key in ("comorbidities", "symptoms", "surgeries", "prior_diagnoses", "current_medications"):
        assert isinstance(h[key], list)


def test_stub_is_seeded_stable():
    a = StubFhirClient().get_patient_history("PT1")
    b = StubFhirClient().get_patient_history("PT1")
    assert a == b


def test_get_fhir_client_default_is_stub():
    assert isinstance(get_fhir_client("stub"), StubFhirClient)


def test_history_drop_in_for_graph_ingestion():
    # the FHIR client output is directly ingestible (same shape the graph ingest expects)
    from kb.graph.ingest import _MANAGES  # noqa: F401 - just proves the module wiring is present

    h = StubFhirClient().get_patient_history("PT9")
    assert set(h) >= {"patient_id", "gender", "age", "comorbidities", "current_medications"}
