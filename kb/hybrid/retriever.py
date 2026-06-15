"""Hybrid retriever — vector passages + graph relationships, side-by-side (D8).

`retrieve(query, mode)` runs the Phase 2A vector search **and** the Phase 2B graph lookup, then
assembles **two clearly labelled blocks** in one `RetrievalResult`:

  * `passages`      — vector chunks (each cited to doc#heading)
  * `relationships` — graph facts (co-morbidity links, protocol/treatment guidance), each cited

There is **no cross-block re-ranking**; the split stays explicit. Under `mode='vector'` the
relationships block is empty (the evaluation baseline).
"""

from __future__ import annotations

from common.schemas import Passage, Relationship, RetrievalResult
from kb.graph.driver import GraphDriver
from kb.graph.templates import run_template
from kb.vector.retriever import VectorRetriever

from .linker import link_conditions


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, driver: GraphDriver) -> None:
        self.vector = vector
        self.driver = driver

    def _condition_names(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        rows = self.driver.run_read(
            "MATCH (c:Condition) WHERE c.id IN $ids RETURN c.id AS id, c.name AS name", ids=ids
        )
        return {r["id"]: (r["name"] or r["id"]) for r in rows}

    def graph_facts(self, query: str) -> list[Relationship]:
        """Traverse from linked conditions into co-morbidity + guidance facts."""
        linked = link_conditions(self.driver, query)
        names = self._condition_names(linked)
        facts: list[Relationship] = []

        for cid in linked:
            cname = names.get(cid, cid)

            for row in run_template(self.driver, "comorbidity_neighborhood", condition_id=cid):
                facts.append(
                    Relationship(
                        fact=(
                            f"{cname} is commonly co-morbid with {row['comorbidity']} "
                            f"(co-occurrence in {row['co_occurrence']} patients, "
                            f"confidence {row['confidence']:.2f})"
                        ),
                        source=row["source"] or "graph:co-morbidity",
                        score=float(row["confidence"]),
                    )
                )

            guidance = run_template(self.driver, "guidance_for_condition", condition_id=cid)
            for row in guidance:
                for title in row["protocols"]:
                    if title:
                        facts.append(
                            Relationship(fact=f"Care protocol '{title}' applies to {cname}",
                                         source=f"protocol:{title}")
                        )
                for g in row["guidelines"]:
                    if g:
                        facts.append(
                            Relationship(fact=f"Clinical guideline applies to {cname}", source=g)
                        )
                for ind in row["indicated_treatments"]:
                    if ind and ind.get("treatment"):
                        facts.append(
                            Relationship(
                                fact=f"{cname} indicates treatment: {ind['treatment']}",
                                source=ind.get("source") or "graph:indicates",
                            )
                        )
        return facts

    def retrieve(self, query: str, mode: str = "hybrid", k: int = 4) -> RetrievalResult:
        passages: list[Passage] = self.vector.retrieve(query, k=k).passages
        relationships = self.graph_facts(query) if mode == "hybrid" else []
        return RetrievalResult(
            query=query, passages=passages, relationships=relationships, mode=mode
        )
