"""Inbound interaction by TEXT (POC option) — the text equivalent of the inbound voice line.

Same PHI-gated path as the phone: authenticate with the shared PIN, then ask grounded questions —
just typed instead of spoken (no STT/TTS). Reuses the Phase 6 `OrchestratorHandler`.

  python -m cli.text_chat --session desk1
  # then: type your 4-digit PIN (default 1234), then a clinical question.
"""

from __future__ import annotations

import argparse

from orchestrator.chat import build_orchestrator
from voice.handlers import OrchestratorHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="text-desk")
    parser.add_argument("--embedder", default="hashing", choices=["auto", "bge", "hashing"])
    parser.add_argument("--llm", default="echo", choices=["echo", "ollama"])
    args = parser.parse_args(argv)

    orch, driver = build_orchestrator(embedder=args.embedder, llm=args.llm, deid="regex")
    handler = OrchestratorHandler(orch, orch.working)
    print(handler.greeting())
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if line.lower() in {"quit", "exit"}:
                break
            if not line:
                continue
            print(handler.respond(line, session_id=args.session))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
