"""Phase 2A vector-RAG CLI.

  python -m cli.kb_vector index   --dir docs/
  python -m cli.kb_vector retrieve "how do I rate control atrial fibrillation"
  python -m cli.kb_vector ask      "what triggers escalation to critical care"

`index` persists into the live Qdrant server; `retrieve`/`ask` query it. Use `--in-memory` to run
a self-contained index+query in one process (no server needed). `--embedder {auto,bge,hashing}`
selects the embedding backend (auto prefers BGE, falls back to hashing offline).
"""

from __future__ import annotations

import argparse
import json

from common.config import DEFAULT
from kb.vector.answer import answer
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore


def _make_store(args: argparse.Namespace) -> QdrantStore:
    if args.in_memory:
        return QdrantStore.in_memory()
    return QdrantStore.connect(args.qdrant_url)


def _make_retriever(args: argparse.Namespace) -> VectorRetriever:
    return VectorRetriever.build(
        store=_make_store(args), embedder_name=args.embedder, rerank=not args.no_rerank
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=DEFAULT.qdrant_url)
    parser.add_argument("--embedder", default="auto", choices=["auto", "bge", "hashing"])
    parser.add_argument("--in-memory", action="store_true", help="self-contained, no server")
    parser.add_argument("--no-rerank", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="chunk + embed + index a corpus")
    p_index.add_argument("--dir", default="docs")

    p_ret = sub.add_parser("retrieve", help="ranked chunks + citations")
    p_ret.add_argument("query")
    p_ret.add_argument("-k", type=int, default=4)

    p_ask = sub.add_parser("ask", help="grounded, cited answer (declines out-of-corpus)")
    p_ask.add_argument("query")
    p_ask.add_argument("-k", type=int, default=4)

    args = parser.parse_args(argv)
    retriever = _make_retriever(args)

    # In-memory mode must index within the same process before querying.
    if args.in_memory and args.cmd in ("retrieve", "ask"):
        retriever.index_dir("docs")

    if args.cmd == "index":
        n = retriever.index_dir(args.dir)
        print(json.dumps({"indexed_chunks": n, "embedder": retriever.embedder.name}))
        return 0

    if args.cmd == "retrieve":
        result = retriever.retrieve(args.query, k=args.k)
        for i, p in enumerate(result.passages, 1):
            print(f"[{i}] score={p.score:.3f}  ({p.source})")
            print(f"    {p.text.replace(chr(10), ' ')[:200]}")
        return 0

    if args.cmd == "ask":
        ans = answer(args.query, retriever, k=args.k)
        print(ans.answer)
        if not ans.declined:
            print("\nCitations: " + ", ".join(ans.citations))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
