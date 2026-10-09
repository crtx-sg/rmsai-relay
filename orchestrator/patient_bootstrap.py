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
_PATIENT_BED = (
    "MATCH (p:Patient {id:$id}) "
    "OPTIONAL MATCH (p)-[r:ASSIGNED_TO]->(b:Bed) WHERE coalesce(r.current, true) "
    "OPTIONAL MATCH (b)-[:IN_UNIT]->(u:Unit) "
    "RETURN p.id AS id, b.label AS bed, coalesce(u.name, split(b.label, '-')[0]) AS unit"
)
_ASSIGN = (
    "MERGE (u:Unit {id:$unit}) SET u.name=$unit "
    "MERGE (b:Bed {id:$bed}) SET b.label=$bed "
    "MERGE (b)-[:IN_UNIT]->(u) "
    "WITH b MATCH (p:Patient {id:$pid}) "
    "MERGE (p)-[r:ASSIGNED_TO]->(b) SET r.current=true"
)


def ensure_patient(driver, beds: BedAssignmentStub, patient_id: str) -> tuple[str, str]:
    """Return (unit, bed) for `patient_id`, creating the patient + history if unknown.

    The graph is the record of who is in which bed; the in-memory stub only picks the next free
    one. Seeded once at startup, the stub went stale (a consumer restart handed a known patient a
    new bed; on 2026-10-09 one patient's events spanned two beds and Bed04 held three patients),
    so the graph is consulted on every call:

    * a patient with a persisted bed keeps it;
    * otherwise the stub is rebuilt from the graph's current assignments and a free bed is picked,
      so it can never hand out a bed the graph shows as taken (even after a graph reset or
      another writer).
    """
    rows = driver.run_read(_PATIENT_BED, id=patient_id)
    if rows and rows[0].get("bed"):
        unit, bed = rows[0]["unit"], rows[0]["bed"]
        beds.claim(patient_id, unit, bed)
        return unit, bed
    beds.clear_all()
    for r in driver.run_read(_CURRENT_BEDS):
        if r.get("bed") and r["id"] != patient_id:
            beds.claim(r["id"], r["unit"], r["bed"])
    beds.seeded = True
    unit, bed = beds.assign(patient_id)
    if not rows:
        history = PatientHistoryStub().get(patient_id).to_dict()
        ingest_patient_record(driver, history, bed=(unit, bed))
    else:  # known patient with no current bed: assign one
        driver.run_write(_ASSIGN, unit=unit, bed=bed, pid=patient_id)
    return unit, bed
