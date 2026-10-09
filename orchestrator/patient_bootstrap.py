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


_CURRENT_BEDS = (
    "MATCH (p:Patient)-[r:ASSIGNED_TO]->(b:Bed) WHERE coalesce(r.current, true) "
    "OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit) "
    "RETURN p.id AS id, b.label AS bed, coalesce(u.name, split(b.label, '-')[0]) AS unit"
)


def ensure_patient(driver, beds: BedAssignmentStub, patient_id: str) -> tuple[str, str]:
    """Return (unit, bed) for `patient_id`, creating the patient + history if unknown.

    The graph is the record of who is in which bed. The stub is in-memory, so after a consumer
    restart it used to hand a known patient a new bed (and a taken bed to someone else): on
    2026-10-09 one patient's events were spread over two beds and Bed04 held three patients.
    A known patient keeps their persisted bed; a new one gets a bed the graph shows as free.
    """
    if not beds.seeded:
        for r in driver.run_read(_CURRENT_BEDS):
            if r.get("bed") and beds.current(r["id"]) is None:
                beds.claim(r["id"], r["unit"], r["bed"])
        beds.seeded = True
    rows = driver.run_read("MATCH (p:Patient {id:$id}) RETURN p.id AS id", id=patient_id)
    unit, bed = beds.assign(patient_id)
    if not rows:
        history = PatientHistoryStub().get(patient_id).to_dict()
        ingest_patient_record(driver, history, bed=(unit, bed))
    return unit, bed
