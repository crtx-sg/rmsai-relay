"""Patient bootstrap (G8): auto-create an unknown patient before persisting an event.

A `DeviceEvent` can arrive for a patient the graph has never seen (a new admission, or a
bus message for a patient ingested on another node). Both the file loop and the bus consumer
need the same behaviour: assign a bed, fetch the synthetic history, and ingest the record so the
event has a `Patient` to link to. References the pseudonym only (G3/G6).
"""

from __future__ import annotations

from common.bed_assignment import BedAssignmentStub
from common.patient_history import PatientHistoryStub
from kb.graph.ingest import ingest_patient_record


def ensure_patient(driver, beds: BedAssignmentStub, patient_id: str) -> tuple[str, str]:
    """Return (unit, bed) for `patient_id`, creating the patient + history if unknown."""
    rows = driver.run_read("MATCH (p:Patient {id:$id}) RETURN p.id AS id", id=patient_id)
    unit, bed = beds.assign(patient_id)
    if not rows:
        history = PatientHistoryStub().get(patient_id).to_dict()
        ingest_patient_record(driver, history, bed=(unit, bed))
    return unit, bed
