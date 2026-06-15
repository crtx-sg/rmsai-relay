"""Document entity extraction (allowlisted) — content graph onto shared nodes.

Extracts `Condition` / `Treatment` / `Guideline` entities and a minimal, **allowlisted** relation
set from the protocol corpus, MERGEing onto the **same** `Condition`/`Treatment` nodes the patient
records use (via `entities.condition_id`/`treatment_id`) so a patient's condition and the same
condition in a protocol resolve to one node. Every extracted edge carries a `source` (doc#chunk)
for citation.

Extraction here is a deterministic dictionary matcher (LLM-assisted extraction can replace the
matcher), but **all writes are constrained to the allowlist** — no arbitrary schema growth.
"""

from __future__ import annotations

from pathlib import Path

from kb.vector.chunking import Chunk, chunk_dir

from .driver import GraphDriver
from .entities import condition_id, slugify, treatment_id

# --- Allowlist: the only labels/edges document extraction may write ---
ALLOWED_LABELS = frozenset({"Condition", "Treatment", "Guideline"})
ALLOWED_EDGES = frozenset(
    {
        ("Guideline", "APPLIES_TO", "Condition"),
        ("Guideline", "RECOMMENDS", "Treatment"),
        ("Condition", "INDICATES", "Treatment"),
    }
)


class AllowlistError(Exception):
    """Raised when extraction tries to write a label/edge outside the allowlist."""


def assert_label_allowed(label: str) -> None:
    if label not in ALLOWED_LABELS:
        raise AllowlistError(f"label {label!r} not in extraction allowlist")


def assert_edge_allowed(src: str, rel: str, dst: str) -> None:
    if (src, rel, dst) not in ALLOWED_EDGES:
        raise AllowlistError(f"edge {(src, rel, dst)} not in extraction allowlist")


# --- Deterministic term dictionaries (canonical names) ---
_CONDITION_TERMS = [
    "atrial fibrillation", "ventricular fibrillation", "ventricular tachycardia",
    "cardiac arrest", "hypotension", "hypertension", "heart failure",
]
_TREATMENT_TERMS = [
    "beta-blocker", "calcium channel blocker", "anticoagulant", "anticoagulation",
    "amiodarone", "epinephrine", "defibrillation", "cardioversion", "rate control",
]


def _find_terms(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t in low]


def extract_chunk(driver: GraphDriver, chunk: Chunk) -> dict:
    """Extract + MERGE entities/edges from one chunk. Returns a small summary."""
    gid = slugify(chunk.source)
    # Guideline node (provenance)
    assert_label_allowed("Guideline")
    driver.run_write(
        "MERGE (g:Guideline {id:$id}) SET g.title=$title, g.source_doc=$doc, g.chunk_ref=$ref",
        id=gid, title=chunk.source, doc=chunk.doc_id, ref=chunk.source,
    )

    conditions = _find_terms(chunk.text, _CONDITION_TERMS)
    treatments = _find_terms(chunk.text, _TREATMENT_TERMS)

    for name in conditions:
        assert_label_allowed("Condition")
        assert_edge_allowed("Guideline", "APPLIES_TO", "Condition")
        driver.run_write(
            "MERGE (c:Condition {id:$cid}) SET c.name=coalesce(c.name,$name) "
            "WITH c MATCH (g:Guideline {id:$gid}) "
            "MERGE (g)-[r:APPLIES_TO]->(c) SET r.source=$src",
            cid=condition_id(name), name=name, gid=gid, src=chunk.source,
        )

    for name in treatments:
        assert_label_allowed("Treatment")
        assert_edge_allowed("Guideline", "RECOMMENDS", "Treatment")
        driver.run_write(
            "MERGE (t:Treatment {id:$tid}) SET t.name=coalesce(t.name,$name) "
            "WITH t MATCH (g:Guideline {id:$gid}) "
            "MERGE (g)-[r:RECOMMENDS]->(t) SET r.source=$src",
            tid=treatment_id(name), name=name, gid=gid, src=chunk.source,
        )

    # INDICATES: condition -> treatment co-mentioned in the same chunk.
    for cname in conditions:
        for tname in treatments:
            assert_edge_allowed("Condition", "INDICATES", "Treatment")
            driver.run_write(
                "MATCH (c:Condition {id:$cid}), (t:Treatment {id:$tid}) "
                "MERGE (c)-[r:INDICATES]->(t) SET r.source=$src",
                cid=condition_id(cname), tid=treatment_id(tname), src=chunk.source,
            )

    return {"guideline": gid, "conditions": conditions, "treatments": treatments}


def extract_dir(driver: GraphDriver, directory: str | Path) -> dict:
    """Extract entities from every document under `directory`. Returns aggregate counts."""
    chunks = chunk_dir(directory)
    summaries = [extract_chunk(driver, c) for c in chunks]
    return {
        "chunks": len(chunks),
        "guidelines": len({s["guideline"] for s in summaries}),
        "conditions": len({c for s in summaries for c in s["conditions"]}),
        "treatments": len({t for s in summaries for t in s["treatments"]}),
    }
