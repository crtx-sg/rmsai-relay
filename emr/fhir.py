"""EMR / FHIR client (S5).

`FhirClient.get_patient_history` returns the same history shape as the `PatientHistory` stub, so
graph ingestion can swap from synthetic data to a real EMR without changing callers:

* `StubFhirClient` — synthetic history (delegates to `PatientHistoryStub`); the POC default.
* `HapiFhirClient` — queries a real HAPI FHIR server (REST, stdlib HTTP, lazy); maps Patient +
  Condition + MedicationRequest + Procedure resources onto our history dict. Verified against the
  `hapi-fhir` compose service (later profile).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod

from common.patient_history import PatientHistoryStub


class FhirClient(ABC):
    @abstractmethod
    def get_patient_history(self, patient_id: str) -> dict:
        """Return {patient_id, gender, age, comorbidities, symptoms, surgeries,
        prior_diagnoses, current_medications}."""


class StubFhirClient(FhirClient):
    """Synthetic patient record (the POC default), seeded by patient_id."""

    def __init__(self) -> None:
        self._stub = PatientHistoryStub()

    def get_patient_history(self, patient_id: str) -> dict:
        return self._stub.get(patient_id).to_dict()


class HapiFhirClient(FhirClient):  # pragma: no cover - needs a live HAPI FHIR server
    """Real EMR via a HAPI FHIR server's REST API (lazy/stdlib)."""

    def __init__(self, base_url: str = "http://localhost:8080/fhir") -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, resource: str, **params) -> dict:
        url = f"{self.base_url}/{resource}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (configured EMR host)
            return json.loads(resp.read())

    @staticmethod
    def _names(bundle: dict, path: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for entry in bundle.get("entry", []):
            node = entry.get("resource", {})
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, str) and node:
                out.append(node)
        return out

    def get_patient_history(self, patient_id: str) -> dict:
        patient_bundle = self._get("Patient", identifier=patient_id)
        patient = (patient_bundle.get("entry") or [{}])[0].get("resource", {})
        birth = patient.get("birthDate", "")
        age = None
        if birth[:4].isdigit():
            age = 2026 - int(birth[:4])  # POC: current-year minus birth-year
        conditions = self._names(self._get("Condition", patient=patient_id),
                                 ("code", "text"))
        meds = self._names(self._get("MedicationRequest", patient=patient_id),
                           ("medicationCodeableConcept", "text"))
        surgeries = self._names(self._get("Procedure", patient=patient_id), ("code", "text"))
        return {
            "patient_id": patient_id,
            "gender": patient.get("gender"),
            "age": age,
            "comorbidities": conditions,
            "symptoms": [],
            "surgeries": surgeries or ["none"],
            "prior_diagnoses": conditions,
            "current_medications": meds,
        }


def get_fhir_client(name: str = "stub", **kwargs) -> FhirClient:
    if name == "hapi":
        return HapiFhirClient(**kwargs)
    return StubFhirClient()
