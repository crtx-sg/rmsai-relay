"""Event history from the graph: every persisted event, filtered by patient, bed, type or status.

  python -m cli.events                                   # every event, newest first
  python -m cli.events --bed Unit1-Bed02 --status pending
  python -m cli.events --bed "bed 2"                     # Bed02 in any unit
  python -m cli.events --patient PT998224 --since 7d
  python -m cli.events --type VENTRICULAR_TACHYCARDIA --labelled
  python -m cli.events --status withheld                 # not alerted (low confidence, FP, ...)
  python -m cli.events --json

  # what a voice/chat question resolves to: the query, its parameters, the rows, the answer
  python -m cli.events --ask "say all unacked events in bed 02"
  python -m cli.events --ask "unacknowledged events for this patient" --as-patient PT998224

  # demo reset: reopen acknowledged/resolved events so re-ingesting them alerts again (writes!)
  python -m cli.events --reopen                          # every closed event
  python -m cli.events --reopen <event-id> <event-id>

`pending` = alerted and not acknowledged/resolved: the same definition the app worklist and the
voice agent use (`kb.graph.events.list_events`), so this listing is how to verify a spoken answer.
Read-only except `--reopen`. Pseudonyms only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from common.config import DEFAULT
from kb.graph.events import STATUS_FILTERS, list_events, reopen_events
from orchestrator.perf_view import parse_since


def _when(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "?"


def _status(r: dict) -> str:
    if r.get("pending"):
        return "PENDING"
    if not r.get("alert_gate"):
        return f"withheld ({r.get('alert_reason_code') or '?'})"
    return r.get("status") or "reported"


def event_lines(rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        delivered = [k for k, v in (("app", r.get("delivered_app")),
                                    (f"call {r.get('delivered_call')}", r.get("delivered_call")),
                                    (r.get("delivered_sms"), r.get("delivered_sms"))) if v]
        line = (f"  {_when(r.get('timestamp'))}  {r.get('patient') or '?':<9} "
                f"{r.get('bed') or '?':<12} {r.get('event_type') or '?':<25} "
                f"{(r.get('confidence') or 0):>4.0%}  {r.get('criticality') or '?':<8} "
                f"{_status(r)}")
        if r.get("alert_basis") == "vitals":
            line += " · vitals alert"
        if delivered:
            line += f" · delivered {', '.join(delivered)}"
        if r.get("ground_truth_condition"):
            line += f" · truth {r['ground_truth_condition']} → {r.get('eval_outcome') or '?'}"
        lines.append(line)
        lines.append(f"      event {r.get('id')}")
    return lines


def _ask(driver, question: str, *, patient_ref: str | None, spoken: bool) -> int:
    """Show how the voice/chat path resolves `question`, without calling any model."""
    from kb.graph.lookup import match_intent  # noqa: PLC0415
    from kb.graph.spoken import normalize_spoken_query  # noqa: PLC0415
    from kb.graph.lookup import looks_like_event_listing  # noqa: PLC0415
    from kb.graph.templates import EVENT_HISTORY, TEMPLATE_DEFAULTS, run_template  # noqa: PLC0415
    from orchestrator.orchestrator import (  # noqa: PLC0415
        _EVENT_LISTING_HINT, _answer_event_history, _answer_operational, _spoken_summary,
    )

    text = normalize_spoken_query(question)
    intent = match_intent(text, now=time.time(), patient_ref=patient_ref)
    print(f"[events] question   {question!r}")
    if text != question:
        print(f"[events] normalized {text!r}")
    if intent is None:
        if looks_like_event_listing(question):
            print(f"[events] declined    {_EVENT_LISTING_HINT}")
        else:
            print("[events] no graph query matches: the agent would not answer this from the graph")
        return 1
    name, params = intent
    shown = {**TEMPLATE_DEFAULTS.get(name, {}), **params}
    print(f"[events] query      {name} {json.dumps(shown, default=str)}")
    rows = run_template(driver, name, **params)
    print(f"[events] rows       {len(rows)}")
    for r in rows:
        print("  " + json.dumps({k: v for k, v in r.items() if v is not None}, default=str))
    if name == EVENT_HISTORY:
        answer = _answer_event_history(rows, params, spoken=spoken)
    else:
        answer = (_spoken_summary(rows, name) if spoken and len(rows) > 1
                  else _answer_operational(rows, name))
    print(f"[events] answer     {answer}")
    return 0


def main(argv: list[str] | None = None, *, driver=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", help="patient pseudonym (PT…)")
    parser.add_argument("--bed", help="bed label (Unit1-Bed02), or Bed02 / 'bed 2' for any unit")
    parser.add_argument("--unit", help="unit name (Unit1)")
    parser.add_argument("--type", dest="event_type", help="event type (VENTRICULAR_TACHYCARDIA)")
    parser.add_argument("--status", choices=STATUS_FILTERS, default="any",
                        help="pending = alerted and not acknowledged/resolved")
    parser.add_argument("--min-criticality", choices=["Low", "Medium", "High", "Critical"])
    parser.add_argument("--since", help="event time: 30m / 24h / 7d, an ISO date, or epoch seconds")
    parser.add_argument("--labelled", action="store_true", help="only events with a ground truth")
    parser.add_argument("--model", help="one model id")
    parser.add_argument("--dataset", help="one source dataset (mitbih, incart, ptbxl, …)")
    parser.add_argument("--limit", type=int, default=None, help="at most N events")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    parser.add_argument("--ask", metavar="QUESTION",
                        help="show which graph query a voice/chat question runs, and its answer")
    parser.add_argument("--as-patient", help="session patient for --ask ('this patient')")
    parser.add_argument("--spoken", action="store_true",
                        help="--ask: render the answer as the phone call would")
    parser.add_argument("--reopen", nargs="*", metavar="EVENT_ID",
                        help="demo reset: set acknowledged/resolved events (all, or these ids) back "
                             "to reported, so a re-ingest alerts again")
    args = parser.parse_args(argv)
    try:
        since = parse_since(args.since)
    except ValueError as exc:
        parser.error(str(exc))

    # Properties an older event never had (delivered_sms, ...) make Neo4j warn on every query.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    if driver is None:
        from kb.graph.driver import GraphDriver  # noqa: PLC0415

        driver = GraphDriver.from_config(DEFAULT)
    if args.reopen is not None:
        n = reopen_events(driver, args.reopen or None)
        print(f"[events] reopened {n} event(s)")
        return 0
    if args.ask:
        return _ask(driver, args.ask, patient_ref=args.as_patient, spoken=args.spoken)

    rows = list_events(
        driver, patient=args.patient, bed=args.bed, unit=args.unit, event_type=args.event_type,
        status=args.status, min_criticality=args.min_criticality, since=since,
        labelled_only=args.labelled, model_id=args.model, dataset=args.dataset, limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0 if rows else 1
    filters = ", ".join(f"{k}={v}" for k, v in (
        ("patient", args.patient), ("bed", args.bed), ("unit", args.unit),
        ("type", args.event_type), ("status", args.status if args.status != "any" else None),
        ("min_criticality", args.min_criticality), ("since", args.since),
        ("labelled", args.labelled or None), ("model", args.model), ("dataset", args.dataset),
    ) if v) or "none"
    pending = sum(1 for r in rows if r["pending"])
    print(f"[events] {len(rows)} event(s), {pending} pending · filters: {filters}")
    if not rows:
        return 1
    print("\n".join(event_lines(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
