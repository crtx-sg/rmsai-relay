"""Patient-record ingestion (MERGE-based, re-ingest-safe).

`ingest_patient_record` MERGEs a patient's demographics + history (conditions, symptoms,
surgeries, medications) and bed/unit assignment into the graph. `derive_comorbidity` rebuilds the
evidence-based `CO_MORBID_WITH` layer from cohort co-occurrence (decision D3) — not from any two
conditions a single patient happens to have.
"""

from __future__ import annotations

from .driver import GraphDriver
from .entities import condition_id, slugify, treatment_id

# Which conditions a medication class manages (for MANAGES edges) — small POC map.
_MANAGES: dict[str, list[str]] = {
    "beta-blocker": ["hypertension", "atrial fibrillation"],
    "ace-inhibitor": ["hypertension", "heart failure"],
    "anticoagulant": ["atrial fibrillation"],
    "statin": ["coronary artery disease"],
    "diuretic": ["heart failure"],
}


def ingest_patient_record(driver: GraphDriver, history: dict, bed: tuple | None = None) -> None:
    """MERGE one patient's demographics, history, and bed/unit assignment (idempotent)."""
    pid = history["patient_id"]
    driver.run_write(
        "MERGE (p:Patient {id:$id}) "
        "SET p.pseudonym=$id, p.gender=$gender, p.age=$age",
        id=pid, gender=history.get("gender"), age=history.get("age"),
    )

    # Conditions: comorbidities + prior diagnoses -> HAS_DIAGNOSIS
    conditions = [c for c in history.get("comorbidities", []) if c]
    conditions += [c for c in history.get("prior_diagnoses", []) if c and c.lower() != "none"]
    for name in conditions:
        driver.run_write(
            "MERGE (c:Condition {id:$cid}) SET c.name=$name "
            "WITH c MATCH (p:Patient {id:$pid}) "
            "MERGE (p)-[r:HAS_DIAGNOSIS]->(c) "
            "SET r.status=coalesce(r.status,'active'), r.severity=coalesce(r.severity,'unknown')",
            cid=condition_id(name), name=name, pid=pid,
        )

    for sym in history.get("symptoms", []):
        if sym:
            driver.run_write(
                "MERGE (s:Symptom {id:$sid}) SET s.name=$name "
                "WITH s MATCH (p:Patient {id:$pid}) MERGE (p)-[:PRESENTS]->(s)",
                sid=slugify(sym), name=sym, pid=pid,
            )

    for surg in history.get("surgeries", []):
        if surg and surg.lower() != "none":
            driver.run_write(
                "MERGE (su:Surgery {id:$sid}) SET su.name=$name "
                "WITH su MATCH (p:Patient {id:$pid}) MERGE (p)-[:HAD_SURGERY]->(su)",
                sid=slugify(surg), name=surg, pid=pid,
            )

    for med in history.get("current_medications", []):
        if not med:
            continue
        driver.run_write(
            "MERGE (t:Treatment {id:$tid}) SET t.name=$name, t.type='medication' "
            "WITH t MATCH (p:Patient {id:$pid}) "
            "MERGE (p)-[r:PRESCRIBED]->(t) SET r.status=coalesce(r.status,'active')",
            tid=treatment_id(med), name=med, pid=pid,
        )
        # MANAGES edges to any conditions present in the graph.
        for cond_name in _MANAGES.get(treatment_id(med).replace("_", "-"), []):
            driver.run_write(
                "MATCH (t:Treatment {id:$tid}), (c:Condition {id:$cid}) MERGE (t)-[:MANAGES]->(c)",
                tid=treatment_id(med), cid=condition_id(cond_name),
            )

    if bed is not None:
        unit, bed_label = bed
        driver.run_write(
            "MERGE (u:Unit {id:$unit}) SET u.name=$unit "
            "MERGE (b:Bed {id:$bed}) SET b.label=$bed "
            "MERGE (b)-[:IN_UNIT]->(u) "
            "WITH b MATCH (p:Patient {id:$pid}) "
            "MERGE (p)-[r:ASSIGNED_TO]->(b) SET r.current=true",
            unit=unit, bed=bed_label, pid=pid,
        )


def derive_comorbidity(driver: GraphDriver, *, min_co_occurrence: int = 2) -> int:
    """Rebuild the CO_MORBID_WITH layer from cohort co-occurrence. Returns #edges created."""
    # Derived layer: drop and rebuild so counts/confidence stay correct as the cohort grows.
    driver.run_write("MATCH ()-[r:CO_MORBID_WITH]->() DELETE r")
    rows = driver.run_write(
        """
        MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(c1:Condition)
        MATCH (p)-[:HAS_DIAGNOSIS]->(c2:Condition)
        WHERE c1.id < c2.id
        WITH c1, c2, count(DISTINCT p) AS cooc
        WHERE cooc >= $min_cooc
        WITH c1, c2, cooc,
             COUNT { (x:Patient)-[:HAS_DIAGNOSIS]->(c1) } AS n1,
             COUNT { (y:Patient)-[:HAS_DIAGNOSIS]->(c2) } AS n2
        WITH c1, c2, cooc, toFloat(cooc) / (n1 + n2 - cooc) AS confidence
        MERGE (c1)-[r:CO_MORBID_WITH]->(c2)
        SET r.co_occurrence_count = cooc,
            r.confidence = confidence,
            r.source = 'cohort-co-occurrence'
        RETURN count(r) AS edges
        """,
        min_cooc=min_co_occurrence,
    )
    return rows[0]["edges"] if rows else 0
