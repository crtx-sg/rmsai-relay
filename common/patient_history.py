"""`PatientHistory` stub (D17).

`get(patient_id)` returns a synthetic history (gender, age, comorbidities, symptoms, surgeries,
prior diagnoses, current medications), **seeded by `patient_id`** so each patient is
distinct-but-stable across calls. A `seed` override yields fresh variants for load/variety tests.

Called once per patient at graph-ingestion time (Phase 2B / G8); all runtime reads come from the
graph, never this stub. The real form is an EMR/FHIR query.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from random import Random

_COMORBIDITIES = [
    "hypertension", "diabetes", "CAD", "heart failure", "COPD", "CKD",
    "atrial fibrillation", "obesity",
]
_SYMPTOMS = ["chest pain", "dyspnea", "palpitations", "dizziness", "syncope", "fatigue"]
_SURGERIES = ["CABG", "PCI/stent", "valve repair", "pacemaker", "none"]
_MEDICATIONS = ["beta-blocker", "ACE-inhibitor", "anticoagulant", "statin", "diuretic"]
_PRIOR_DX = ["MI", "stroke", "arrhythmia", "valvular disease", "none"]


@dataclass
class PatientHistory:
    patient_id: str
    gender: str
    age: int
    comorbidities: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    surgeries: list[str] = field(default_factory=list)
    prior_diagnoses: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "gender": self.gender,
            "age": self.age,
            "comorbidities": self.comorbidities,
            "symptoms": self.symptoms,
            "surgeries": self.surgeries,
            "prior_diagnoses": self.prior_diagnoses,
            "current_medications": self.current_medications,
        }


def _seed_for(patient_id: str, seed: int | None) -> int:
    base = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    n = int(base[:16], 16)
    return n if seed is None else n ^ (seed * 0x9E3779B97F4A7C15)


def _subset(rng: Random, pool: list[str], lo: int, hi: int) -> list[str]:
    k = rng.randint(lo, min(hi, len(pool)))
    return sorted(rng.sample(pool, k))


class PatientHistoryStub:
    """Deterministic synthetic history service, seeded by patient_id."""

    def get(self, patient_id: str, seed: int | None = None) -> PatientHistory:
        rng = Random(_seed_for(patient_id, seed))
        return PatientHistory(
            patient_id=patient_id,
            gender=rng.choice(["M", "F"]),
            age=rng.randint(18, 95),
            comorbidities=_subset(rng, _COMORBIDITIES, 0, 3),
            symptoms=_subset(rng, _SYMPTOMS, 0, 3),
            surgeries=_subset(rng, _SURGERIES, 1, 2),
            prior_diagnoses=_subset(rng, _PRIOR_DX, 1, 2),
            current_medications=_subset(rng, _MEDICATIONS, 0, 3),
        )
