"""Load curated care protocols into the graph (D19).

Parses the YAML/JSON protocol config (via `common.protocol_loader`) and MERGEs
`CareProtocol`/`ProtocolStep` nodes, links `APPLIES_TO` the shared `Condition` node, `HAS_STEP`
the ordered steps, and `RECOMMENDS` a `Treatment` for medication steps. The graph is the only store:
`orchestrator.protocol_answer` reads these steps verbatim; protocol *documents* (`docs/`, uploads)
are what the vector store indexes.
"""

from __future__ import annotations

from pathlib import Path

from common.protocol_loader import CareProtocol, load_protocols

from .driver import GraphDriver
from .entities import condition_id, treatment_id


def _event_type_to_condition_id(event_type: str) -> str | None:
    if not event_type or event_type == "*":
        return None
    return condition_id(event_type.replace("_", " "))


def load_protocol(driver: GraphDriver, proto: CareProtocol) -> None:
    driver.run_write(
        "MERGE (cp:CareProtocol {id:$id}) "
        "SET cp.title=$title, cp.version=$version, cp.source=$source, "
        "cp.event_type=$etype, cp.min_severity=$sev",
        id=proto.id, title=proto.title, version=proto.version, source=proto.source,
        etype=proto.match.event_type, sev=proto.match.min_severity,
    )

    cond_id = _event_type_to_condition_id(proto.match.event_type)
    if cond_id:
        driver.run_write(
            "MERGE (c:Condition {id:$cid}) "
            "WITH c MATCH (cp:CareProtocol {id:$id}) MERGE (cp)-[:APPLIES_TO]->(c)",
            cid=cond_id, id=proto.id,
        )

    for step in proto.steps:
        sid = f"{proto.id}_step_{step.order}"
        driver.run_write(
            "MATCH (cp:CareProtocol {id:$pid}) "
            "MERGE (s:ProtocolStep {id:$sid}) "
            "SET s.order=$order, s.kind=$kind, s.text=$text "
            "MERGE (cp)-[:HAS_STEP]->(s)",
            pid=proto.id, sid=sid, order=step.order, kind=step.kind, text=step.render(),
        )
        # Medication steps recommend a Treatment (shared node).
        if step.kind == "medication":
            drug = step.fields.get("drug")
            if drug:
                driver.run_write(
                    "MERGE (t:Treatment {id:$tid}) SET t.name=coalesce(t.name,$name) "
                    "WITH t MATCH (s:ProtocolStep {id:$sid}) MERGE (s)-[:RECOMMENDS]->(t)",
                    tid=treatment_id(drug), name=drug, sid=sid,  # synonyms fold: "ACE-I" = the patient's node
                )


def load_protocol_file(driver: GraphDriver, path: str | Path) -> int:
    """Make the graph's protocols exactly the file's (the hospital's set replaces any earlier one):
    MERGE each, then delete protocols and steps the file no longer has, so a removed protocol is
    never read out. Shared Condition/Treatment nodes stay. Returns #protocols loaded."""
    protocols = load_protocols(path)
    if not protocols:  # an empty/half-saved file must not wipe the hospital's protocols
        raise ValueError(f"no protocols in {path}: refusing to replace the graph's set with nothing")
    for proto in protocols:
        load_protocol(driver, proto)
    ids = [p.id for p in protocols]
    step_ids = [f"{p.id}_step_{s.order}" for p in protocols for s in p.steps]
    driver.run_write(
        "MATCH (cp:CareProtocol) WHERE NOT cp.id IN $ids "
        "OPTIONAL MATCH (cp)-[:HAS_STEP]->(s:ProtocolStep) DETACH DELETE s, cp", ids=ids)
    driver.run_write(  # a kept protocol that lost steps
        "MATCH (:CareProtocol)-[:HAS_STEP]->(s:ProtocolStep) WHERE NOT s.id IN $sids "
        "DETACH DELETE s", sids=step_ids)
    return len(protocols)
