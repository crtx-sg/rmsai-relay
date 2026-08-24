"""Show how a question routes: regex template, LLM-routed template, or document retrieval.

The three paths answer very differently — a graph template gives exact patient data, document
retrieval gives cited protocol text — so "why did I get that answer?" is usually really "which path
did my question take?". This answers that without running a full turn (no KB, no orchestrator).

  # what the regexes alone do with it
  uv run python -m cli.kb_route "how is the patient in Unit1-Bed01 doing?"

  # ...and what the LLM router makes of it when they miss (needs LLM_PROVIDER=ollama)
  uv run python -m cli.kb_route --llm "any critical events overnight?"

  # with session scope, as the companion app would have it
  uv run python -m cli.kb_route --llm --patient PT1155 "how have their vitals been?"
"""

from __future__ import annotations

import argparse
import time

from common.config import DEFAULT
from kb.graph.llm_router import build_prompt, looks_operational
from kb.graph.llm_router import route as llm_route
from kb.graph.lookup import match_intent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--llm", action="store_true", help="Try the LLM router when regexes miss.")
    parser.add_argument("--patient", default=None, help="Session patient (e.g. PT1155).")
    parser.add_argument("--event", default=None, help="Selected event uuid.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the routing prompt.")
    args = parser.parse_args(argv)

    now = time.time()
    intent = match_intent(args.query, now=now, patient_ref=args.patient, event_ref=args.event)
    if intent:
        print(f"[route] regex -> {intent[0]}  params={intent[1]}")
        return 0
    print("[route] regex -> (no match)")

    operational = looks_operational(args.query)
    print(f"[route] looks operational? {operational} "
          f"({'the LLM router is eligible' if operational else 'document retrieval only'})")
    if args.show_prompt:
        print("--- routing prompt ---")
        print(build_prompt(args.query, has_patient=bool(args.patient), has_event=bool(args.event)))
        print("--- end ---")
    if not args.llm:
        print("[route] pass --llm to try the LLM router")
        return 0
    if not operational:
        print("[route] llm -> skipped (question is not about patients/monitoring)")
        return 0

    from common.deid import get_deidentifier  # noqa: PLC0415
    from common.providers import DeidentifyingLLM, get_llm_provider  # noqa: PLC0415

    llm = DeidentifyingLLM(get_llm_provider(DEFAULT.llm_provider, DEFAULT), get_deidentifier())
    started = time.time()
    routed = llm_route(args.query, llm, patient_ref=args.patient, event_ref=args.event, now=now)
    elapsed = time.time() - started
    if routed:
        print(f"[route] llm -> {routed[0]}  params={routed[1]}  ({elapsed:.1f}s, "
              f"provider={DEFAULT.llm_provider}/{DEFAULT.llm_model})")
    else:
        print(f"[route] llm -> (no match; falls through to document retrieval) ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
