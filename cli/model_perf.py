"""Model-performance summary on request, from the graph (all labelled events ever persisted).

  python -m cli.model_perf                       # every labelled event, one block per model
  python -m cli.model_perf --since 24h           # processed in the last 24 h (also 30m, 7d, ISO, epoch)
  python -m cli.model_perf --dataset mitbih      # one source dataset
  python -m cli.model_perf --model stub          # one model id
  python -m cli.model_perf --events              # also list each event: outcome, source, why
  python -m cli.model_perf --json                # machine-readable

Outcomes and metrics come from `inference.metrics` (positive = any arrhythmia; a wrong-class
arrhythmia is TP_WRONG_CLASS; rates with Wilson 95 % intervals). Events are grouped by model,
because a summary mixing two models' predictions describes neither. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from common.config import DEFAULT
from inference.metrics import format_summary, summarize
from orchestrator.perf_view import group_by_model, parse_since, records, row_source

# Re-exported for callers/tests that import them from here.
__all__ = ["group_by_model", "parse_since", "report", "main"]


def event_lines(rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        stamp = r.get("processed_at") or r.get("timestamp")
        when = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M") if stamp else "?"
        lines.append(f"  {when} {r['patient']:<9} {r.get('eval_outcome') or '?':<15} "
                     f"pred {r['event_type']} ({(r.get('confidence') or 0):.0%}) · truth "
                     f"{r.get('ground_truth_condition') or '—'} · {row_source(r)}")
        if r.get("why"):
            lines.append(f"      why: {r['why']}")
    return lines


def report(rows: list[dict], *, events: bool = False) -> tuple[str, dict]:
    """Text report and the JSON structure, one block per model."""
    blocks, data = [], {}
    for model, m in group_by_model(rows).items():
        s = summarize(records(m["rows"]), m["classes"])
        data[model] = s
        block = [f"== model {model}", format_summary(s)]
        datasets = sorted({r.get("source_dataset") or r.get("source_kind") or "unknown"
                           for r in m["rows"]})
        block.append(f"  sources      {', '.join(datasets)}")
        if events:
            block.append("  events (most recently processed first; time = processed):")
            block.extend(event_lines(m["rows"]))
        blocks.append("\n".join(block))
    return "\n\n".join(blocks), data


def main(argv: list[str] | None = None, *, driver=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="30m / 24h / 7d, an ISO date, or epoch seconds")
    parser.add_argument("--model", help="one model id (see the '== model' headers)")
    parser.add_argument("--dataset", help="one source dataset (mitbih, incart, ptbxl, …)")
    parser.add_argument("--events", action="store_true", help="list each event with outcome + why")
    parser.add_argument("--json", action="store_true", help="print the summaries as JSON")
    args = parser.parse_args(argv)
    try:
        since = parse_since(args.since)
    except ValueError as exc:
        parser.error(str(exc))

    from kb.graph.events import eval_events  # noqa: PLC0415

    if driver is None:
        from kb.graph.driver import GraphDriver  # noqa: PLC0415

        driver = GraphDriver.from_config(DEFAULT)
    rows = eval_events(driver, since=since, model_id=args.model, dataset=args.dataset)
    if not rows:
        print("no labelled events match (production data has no ground truth; curate a labelled "
              "set with cli.real_samples, or relax the filters)", file=sys.stderr)
        return 1
    text, data = report(rows, events=args.events)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        filters = ", ".join(f"{k}={v}" for k, v in (("since", args.since), ("model", args.model),
                                                    ("dataset", args.dataset)) if v) or "none"
        print(f"[model_perf] {len(rows)} labelled event(s) · filters: {filters}\n")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
