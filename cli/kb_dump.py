"""Dump everything stored for one event — graph node + neighbors AND vector report chunks.

  python -m cli.kb_dump <event_id>           # human-readable, side by side
  python -m cli.kb_dump <event_id> --json    # raw {graph, vector} for tooling
  python -m cli.kb_dump --list               # list recent event ids to pick from

Two stores, one event: the graph (Neo4j) is the structured source of truth — a `MonitoredEvent`
with the vitals snapshot and links to patient/bed/condition/action/report; the vector store (Qdrant)
holds the report *narrative* indexed for semantic Q&A under `doc_id=report:<event_id>`. This harness
shows both so you can see exactly what was persisted where.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import DEFAULT
from kb.graph.driver import GraphDriver
from kb.vector.store import QdrantStore

_EVENT_CYPHER = """
MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent {id:$id})
OPTIONAL MATCH (e)-[:AT_BED]->(b:Bed)
OPTIONAL MATCH (e)-[:OF_CONDITION]->(c:Condition)
OPTIONAL MATCH (e)-[:HAS_ACTION]->(a:ActionItem)
OPTIONAL MATCH (e)-[:HAS_REPORT]->(r:Report)
RETURN p.id AS patient, properties(e) AS event, b.label AS bed,
       collect(DISTINCT c.name) AS conditions,
       collect(DISTINCT properties(a)) AS actions,
       properties(r) AS report
"""

_LIST_CYPHER = """
MATCH (p:Patient)-[:HAD_EVENT]->(e:MonitoredEvent)
RETURN p.id AS patient, e.id AS event_id, e.event_type AS type, e.criticality AS crit
ORDER BY e.timestamp DESC LIMIT $limit
"""


def fetch_event(driver, event_id: str) -> dict | None:
    rows = driver.run_read(_EVENT_CYPHER, id=event_id)
    return rows[0] if rows else None


def _fmt_vitals(ev: dict) -> str:
    g = ev.get
    return (f"HR {g('hr')}, BP {g('sbp')}/{g('dbp')}, SpO2 {g('spo2')}, "
            f"RR {g('rr')}, Temp {g('temp')}")


def render_dump(event_id: str, graph: dict | None, chunks: list[dict],
                report_text: str | None = None) -> str:
    """Pure renderer: format graph row + report file + vector chunks into a readable dump."""
    out = [f"=== EVENT {event_id} ===", ""]

    out.append("GRAPH (Neo4j)")
    if graph is None:
        out.append("  (no MonitoredEvent with this id)")
    else:
        ev = graph.get("event") or {}
        rep = graph.get("report") or {}
        actions = [a.get("text") for a in (graph.get("actions") or []) if a]
        out += [
            f"  patient    : {graph.get('patient')}",
            f"  bed        : {graph.get('bed')}",
            f"  event_type : {ev.get('event_type')}  | criticality {ev.get('criticality')}"
            f"  | status {ev.get('status')}  | FP {ev.get('is_false_positive')}",
            f"  confidence : {ev.get('confidence')}  | MEWS risk {ev.get('mews_risk')}",
            f"  vitals     : {_fmt_vitals(ev)}",
            f"  conditions : {', '.join(graph.get('conditions') or []) or '(none)'}",
            f"  actions    : {', '.join(actions) or '(none)'}",
            f"  signal_ref : {ev.get('signal_ref')}",
            f"  plots      : ecg={ev.get('ecg_plot_ref')}  vitals={ev.get('vitals_plot_ref')}",
        ]
        if rep:
            uri = rep.get("uri")
            exists = report_text is not None
            out += [
                f"  report     : {rep.get('id')}  (index_status {rep.get('index_status')})",
                f"  report uri : {uri}  ({'exists' if exists else 'MISSING on disk'})",
                f"               summary: {rep.get('summary')!r}",
            ]
    out.append("")

    if report_text is not None:
        out.append("REPORT FILE (markdown)")
        out += [f"  {line}" for line in report_text.rstrip().splitlines()]
        out.append("")

    out.append(f"VECTOR (Qdrant)  doc_id=report:{event_id}")
    if not chunks:
        out.append("  chunks: 0  (not indexed, or wiped by a `kb_vector index --reset` rebuild)")
    else:
        out.append(f"  chunks: {len(chunks)}")
        for i, ch in enumerate(chunks, 1):
            text = (ch.get("text") or "").replace("\n", " ")
            out.append(f"  [{i}] ({ch.get('source')})")
            out.append(f"      {text[:200]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event_id", nargs="?", help="MonitoredEvent uuid to dump")
    parser.add_argument("--list", action="store_true", help="list recent event ids and exit")
    parser.add_argument("--json", action="store_true", help="emit raw {graph, vector} JSON")
    parser.add_argument("--limit", type=int, default=20, help="rows for --list")
    args = parser.parse_args(argv)

    with GraphDriver.from_config(DEFAULT) as driver:
        if args.list or not args.event_id:
            rows = driver.run_read(_LIST_CYPHER, limit=args.limit)
            print(json.dumps(rows, default=str, indent=2))
            return 0

        graph = fetch_event(driver, args.event_id)
        store = QdrantStore.connect(DEFAULT.qdrant_url)
        chunks = store.chunks_for_doc(f"report:{args.event_id}")

    # Read the materialized report file if the graph points at one that exists on disk.
    report_text = None
    rep = (graph or {}).get("report") or {}
    if rep.get("uri"):
        p = Path(rep["uri"])
        if p.is_file():
            report_text = p.read_text(encoding="utf-8")

    if args.json:
        print(json.dumps({"graph": graph, "vector": chunks, "report_text": report_text},
                         default=str, indent=2))
    else:
        print(render_dump(args.event_id, graph, chunks, report_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
