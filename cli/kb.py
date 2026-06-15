"""Phase 2C hybrid KB CLI.

  python -m cli.kb ask "which conditions are commonly co-morbid with atrial fibrillation"
  python -m cli.kb ask --mode vector "..."     # baseline: passages only

Runs the vector search (live Qdrant) + graph lookup (live Neo4j), assembles the two labelled
blocks, and answers grounded in both. `--show-context` prints the blocks.
"""

from __future__ import annotations

import argparse

from common.config import DEFAULT
from kb.graph.driver import GraphDriver
from kb.hybrid.answer import answer
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    parser.add_argument("--embedder", default="auto", choices=["auto", "bge", "hashing"])
    parser.add_argument("--in-memory", action="store_true", help="self-contained vector index")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args(argv)

    store = QdrantStore.in_memory() if args.in_memory else QdrantStore.connect(DEFAULT.qdrant_url)
    vector = VectorRetriever.build(store=store, embedder_name=args.embedder)
    if args.in_memory:
        vector.index_dir(args.docs)

    with GraphDriver.from_config(DEFAULT) as driver:
        retriever = HybridRetriever(vector, driver)
        if args.show_context:
            result = retriever.retrieve(args.query, mode=args.mode, k=args.k)
            print("## Retrieved passages")
            for p in result.passages:
                print(f"  - ({p.source}) {p.text.replace(chr(10), ' ')[:120]}")
            print("## Known relationships")
            for r in result.relationships:
                print(f"  - ({r.source}) {r.fact}")
            print()

        ans = answer(args.query, retriever, mode=args.mode, k=args.k)
        print(ans.answer)
        if not ans.declined:
            print("\nCitations: " + ", ".join(ans.citations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
