"""Phase 4 text-chat REPL.

  python -m orchestrator.chat --session demo --patient PT1155

Wires the orchestrator over live backends (Redis working memory, Qdrant vector + episodic, Neo4j
graph) with a de-identifying LLM (EchoLLM by default; --llm ollama for a real local model).
"""

from __future__ import annotations

import argparse

from common.config import DEFAULT
from common.deid import get_deidentifier
from common.providers import DeidentifyingLLM, get_llm_provider
from kb.graph.driver import GraphDriver
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory

from .orchestrator import Orchestrator


def build_orchestrator(
    *, embedder: str | None = None, llm: str | None = None, deid: str | None = None,
    docs: str = "docs", config=DEFAULT,
) -> tuple[Orchestrator, GraphDriver]:
    """Wire the orchestrator over live backends. Unset args fall back to `config` (env)."""
    embedder = embedder or config.embedder
    deid = deid or config.deid_backend
    vector = VectorRetriever.build(
        store=QdrantStore.connect(config.qdrant_url, "rmsai_docs"), embedder_name=embedder
    )
    vector.index_dir(docs)
    driver = GraphDriver.from_config(config)
    llm_provider = get_llm_provider(llm or config.llm_provider, config)
    orch = Orchestrator(
        working=WorkingMemory.from_config(),
        hybrid=HybridRetriever(vector, driver),
        episodic=EpisodicMemory.from_config(embedder_name=embedder),
        llm=DeidentifyingLLM(llm_provider, get_deidentifier(deid)),
        driver=driver,
    )
    return orch, driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="repl")
    parser.add_argument("--patient", default=None)
    parser.add_argument("--embedder", default="hashing", choices=["auto", "bge", "hashing"])
    parser.add_argument("--llm", default="echo", choices=["echo", "ollama"])
    parser.add_argument("--deid", default="regex", choices=["regex", "auto", "presidio"])
    args = parser.parse_args(argv)

    orch, driver = build_orchestrator(embedder=args.embedder, llm=args.llm, deid=args.deid)
    print("rmsai chat — type a question, or 'quit' to exit.")
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            result = orch.handle_turn(args.session, text, patient_ref=args.patient)
            print(result.answer)
            if result.citations and not result.declined:
                print("  citations: " + ", ".join(result.citations))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
