"""Phase 3 memory demo CLI — round-trips each tier in isolation.

  python -m cli.memory demo            # working (Redis) + episodic (Qdrant)

Working memory needs Redis; episodic needs Qdrant. (Semantic recall is the KB itself: `cli.kb`.)
"""

from __future__ import annotations

import argparse
import time

from common.schemas import ChatTurn


def _demo() -> int:
    from memory.episodic import EpisodicMemory
    from memory.working import WorkingMemory

    sid = f"demo-{int(time.time())}"

    print("== working memory (Redis) ==")
    wm = WorkingMemory.from_config()
    wm.append_turn(sid, ChatTurn(role="clinician", text="status of bed 3?", timestamp=1.0))
    wm.append_turn(sid, ChatTurn(role="assistant", text="VT detected at 14:02", timestamp=2.0))
    state = WorkingMemory.from_config().load(sid)  # fresh instance = cross-process
    print(f"  session {sid}: {[t.text for t in state.turns]}")
    wm.clear(sid)

    print("== episodic memory (Qdrant) ==")
    em = EpisodicMemory.from_config(embedder_name="hashing", collection="rmsai_episodic_demo")
    em.add("clinician acknowledged the VT alert on bed 3", patient_ref="PT1")
    em.add("discussed afib rate control with beta-blockers", patient_ref="PT1")
    hits = em.recall("what did we decide about atrial fibrillation rate control", patient_ref="PT1")
    print(f"  recalled: {hits[0].text!r} (score={hits[0].score:.3f})" if hits else "  (none)")
    em.client.delete_collection("rmsai_episodic_demo")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="round-trip the working and episodic tiers")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
