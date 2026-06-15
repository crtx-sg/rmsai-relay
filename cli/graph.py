"""Phase 2B graph CLI.

  python -m cli.graph migrate
  python -m cli.graph ingest --patients PT1000 PT1001 PT1002    # synthetic cohort -> graph
  python -m cli.graph extract --dir docs                        # document entities -> shared nodes
  python -m cli.graph protocols                                 # load care protocols
  python -m cli.graph lookup "critical events in the last 24 hours"
  python -m cli.graph template outstanding_action_items
"""

from __future__ import annotations

import argparse
import json
import time

from cli.gen_synthetic import build_cohort
from common.config import DEFAULT
from kb.graph.driver import GraphDriver
from kb.graph.extract import extract_dir
from kb.graph.ingest import derive_comorbidity, ingest_patient_record
from kb.graph.lookup import lookup
from kb.graph.protocols import load_protocol_file
from kb.graph.schema import migrate
from kb.graph.templates import TEMPLATES, run_template

_PROTOCOLS = "common/protocols/care_protocols.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate", help="create constraints + indexes")

    p_ing = sub.add_parser("ingest", help="ingest a synthetic patient cohort")
    p_ing.add_argument("--patients", nargs="+", required=True)

    p_ext = sub.add_parser("extract", help="extract document entities onto shared nodes")
    p_ext.add_argument("--dir", default="docs")

    sub.add_parser("protocols", help="load care protocols into the graph")

    p_look = sub.add_parser("lookup", help="natural-language operational query")
    p_look.add_argument("query")

    p_tpl = sub.add_parser("template", help="run a named template")
    p_tpl.add_argument("name", choices=sorted(TEMPLATES))
    p_tpl.add_argument("--param", action="append", default=[], help="k=v (repeatable)")

    args = parser.parse_args(argv)

    with GraphDriver.from_config(DEFAULT) as driver:
        if args.cmd == "migrate":
            migrate(driver)
            print(json.dumps({"migrated": True}))
        elif args.cmd == "ingest":
            cohort = build_cohort(args.patients)
            for entry in cohort:
                ingest_patient_record(driver, entry["history"], bed=(entry["unit"], entry["bed"]))
            edges = derive_comorbidity(driver)
            print(json.dumps({"patients": len(cohort), "comorbidity_edges": edges}))
        elif args.cmd == "extract":
            print(json.dumps(extract_dir(driver, args.dir)))
        elif args.cmd == "protocols":
            print(json.dumps({"protocols_loaded": load_protocol_file(driver, _PROTOCOLS)}))
        elif args.cmd == "lookup":
            result = lookup(driver, args.query, now=time.time())
            print(json.dumps(result, default=str, indent=2))
        elif args.cmd == "template":
            params = dict(p.split("=", 1) for p in args.param)
            for k, v in params.items():
                if v.replace(".", "", 1).lstrip("-").isdigit():
                    params[k] = float(v) if "." in v else int(v)
            print(json.dumps(run_template(driver, args.name, **params), default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
