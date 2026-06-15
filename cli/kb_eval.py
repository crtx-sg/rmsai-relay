"""Phase 2D evaluation CLI: vector vs hybrid over the gold question set.

  python -m cli.kb_eval                       # seeds the eval graph + indexes docs, runs both modes
  python -m cli.kb_eval --questions kb/eval/questions.json --json

Reports correctness, citation grounding, context-token cost, and latency side by side — the
go/no-go on hybrid for the orchestrator.
"""

from __future__ import annotations

import argparse
import json

from common.config import DEFAULT
from kb.eval.fixtures import seed_eval_graph
from kb.eval.harness import load_questions, render_report, run_eval
from kb.graph.driver import GraphDriver
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore

_DEFAULT_QUESTIONS = "kb/eval/questions.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=_DEFAULT_QUESTIONS)
    parser.add_argument("--embedder", default="hashing", choices=["auto", "bge", "hashing"])
    parser.add_argument("--no-seed", action="store_true", help="use the graph as-is (don't reseed)")
    parser.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name=args.embedder)
    vector.index_dir("docs")

    with GraphDriver.from_config(DEFAULT) as driver:
        if not args.no_seed:
            seed_eval_graph(driver)
        report = run_eval(HybridRetriever(vector, driver), questions)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
