"""Query → graph entity linking.

Finds which `Condition` nodes a natural-language query refers to, so the hybrid retriever can
traverse from them into relationships (co-morbidity, protocol/treatment guidance). Matches the
query against actual condition names in the graph plus the canonical synonym map (so "afib"
links to the `atrial_fibrillation` node).
"""

from __future__ import annotations

from kb.graph.driver import GraphDriver
from kb.graph.entities import _CONDITION_SYNONYMS, condition_id


def link_conditions(driver: GraphDriver, query: str) -> list[str]:
    """Return the ids of Condition nodes mentioned in the query (order-stable, deduped)."""
    q = query.lower()
    linked: list[str] = []

    # 1. Match against actual condition names present in the graph.
    rows = driver.run_read("MATCH (c:Condition) RETURN c.id AS id, c.name AS name")
    for r in rows:
        name = (r["name"] or "").lower()
        if name and name in q and r["id"] not in linked:
            linked.append(r["id"])

    # 2. Synonym surface forms ("afib", "chf", ...) -> canonical condition id.
    known_ids = {r["id"] for r in rows}
    for surface, canonical in _CONDITION_SYNONYMS.items():
        if surface in q:
            cid = condition_id(canonical)
            if cid in known_ids and cid not in linked:
                linked.append(cid)

    return linked
