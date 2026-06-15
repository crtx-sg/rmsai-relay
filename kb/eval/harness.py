"""Evaluation harness: vector vs hybrid over one question set.

Runs each gold question through both modes and reports, per mode:
  * correctness   — answer contains the expected facts (or correctly declines)
  * grounding     — citations trace to the required block (graph / doc)
  * context cost  — approximate tokens of the assembled context
  * latency       — wall-clock per query

The point is to quantify what the graph adds (relationship questions) and at what cost
(extra context + latency) — the go/no-go on hybrid for the orchestrator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from kb.hybrid.answer import answer
from kb.hybrid.retriever import HybridRetriever

MODES = ("vector", "hybrid")


@dataclass
class Question:
    id: str
    type: str  # relationship | passage | safety
    question: str
    behavior: str  # answer | decline
    expect_facts: list[str] = field(default_factory=list)
    must_cite: str | None = None  # graph | doc | None


def load_questions(path: str | Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(**item) for item in data]


def _approx_tokens(result) -> int:
    chars = sum(len(p.text) for p in result.passages) + sum(len(r.fact) for r in result.relationships)
    return chars // 4


def _is_graph_citation(c: str) -> bool:
    return "co-occurrence" in c or c.startswith("protocol:") or "graph:" in c


def _is_doc_citation(c: str) -> bool:
    return ".md" in c


def _score_correct(q: Question, ans) -> bool:
    if q.behavior == "decline":
        return ans.declined
    if ans.declined:
        return False
    text = ans.answer.lower()
    return all(f.lower() in text for f in q.expect_facts)


def _score_grounded(q: Question, ans) -> bool | None:
    if q.behavior == "decline":
        return None  # grounding not applicable to a (correct) decline
    if ans.declined or not ans.citations:
        return False
    if q.must_cite == "graph":
        return any(_is_graph_citation(c) for c in ans.citations)
    if q.must_cite == "doc":
        return any(_is_doc_citation(c) for c in ans.citations)
    return True


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_eval(retriever: HybridRetriever, questions: list[Question]) -> dict:
    """Run every question through both modes; return a structured report."""
    report: dict = {"modes": {}, "questions": len(questions)}
    for mode in MODES:
        items = []
        for q in questions:
            result = retriever.retrieve(q.question, mode=mode)  # for cost (untimed)
            t0 = time.perf_counter()
            ans = answer(q.question, retriever, mode=mode)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            items.append(
                {
                    "id": q.id, "type": q.type, "behavior": q.behavior,
                    "correct": _score_correct(q, ans),
                    "grounded": _score_grounded(q, ans),
                    "tokens": _approx_tokens(result),
                    "latency_ms": latency_ms,
                    "declined": ans.declined,
                }
            )
        grounded_vals = [i["grounded"] for i in items if i["grounded"] is not None]
        by_type: dict[str, float] = {}
        for t in {q.type for q in questions}:
            t_items = [i for i in items if i["type"] == t]
            by_type[t] = _mean([1.0 if i["correct"] else 0.0 for i in t_items])
        report["modes"][mode] = {
            "items": items,
            "correctness": _mean([1.0 if i["correct"] else 0.0 for i in items]),
            "grounding": _mean([1.0 if g else 0.0 for g in grounded_vals]),
            "avg_tokens": _mean([i["tokens"] for i in items]),
            "avg_latency_ms": _mean([i["latency_ms"] for i in items]),
            "correctness_by_type": by_type,
        }
    return report


def render_report(report: dict) -> str:
    lines = [f"KB evaluation — {report['questions']} questions, vector vs hybrid", ""]
    header = f"{'metric':<26}{'vector':>12}{'hybrid':>12}"
    lines += [header, "-" * len(header)]
    v, h = report["modes"]["vector"], report["modes"]["hybrid"]

    def row(label, vv, hv, pct=False, ms=False):
        fmt = (lambda x: f"{x * 100:.0f}%") if pct else ((lambda x: f"{x:.1f}ms") if ms else (lambda x: f"{x:.1f}"))
        lines.append(f"{label:<26}{fmt(vv):>12}{fmt(hv):>12}")

    row("correctness (overall)", v["correctness"], h["correctness"], pct=True)
    for t in sorted(v["correctness_by_type"]):
        row(f"  correctness: {t}", v["correctness_by_type"][t], h["correctness_by_type"][t], pct=True)
    row("citation grounding", v["grounding"], h["grounding"], pct=True)
    row("avg context tokens", v["avg_tokens"], h["avg_tokens"])
    row("avg latency", v["avg_latency_ms"], h["avg_latency_ms"], ms=True)

    lines += ["", "Per-question (correct? / mode):", f"{'id':<5}{'type':<14}{'vector':>8}{'hybrid':>8}"]
    for vi, hi in zip(v["items"], h["items"]):
        ok = lambda b: "ok" if b else "MISS"  # noqa: E731
        lines.append(f"{vi['id']:<5}{vi['type']:<14}{ok(vi['correct']):>8}{ok(hi['correct']):>8}")
    return "\n".join(lines)
