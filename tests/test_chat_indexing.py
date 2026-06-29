"""build_orchestrator must APPEND docs (reset=False), never reset Qdrant.

It runs on every orchestrator build — inbound text chat and every voice call — so a reset here
would wipe the event-report narratives that `consume` archived into the same collection. Backends
are monkeypatched so this stays offline (no Qdrant/Neo4j/Redis).
"""

from __future__ import annotations

from orchestrator import chat as chatmod


def test_build_orchestrator_appends_not_resets(monkeypatch):
    seen = {}

    class _FakeVector:
        def index_dir(self, docs, *, reset=True):
            seen["docs"] = docs
            seen["reset"] = reset
            return 0

    # Replace every live collaborator with a no-op so build_orchestrator wires without infra.
    monkeypatch.setattr(chatmod.QdrantStore, "connect", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(chatmod.VectorRetriever, "build", staticmethod(lambda **k: _FakeVector()))
    monkeypatch.setattr(chatmod.GraphDriver, "from_config", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(chatmod.WorkingMemory, "from_config", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(chatmod.EpisodicMemory, "from_config", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(chatmod, "get_llm_provider", lambda *a, **k: object())
    monkeypatch.setattr(chatmod, "get_deidentifier", lambda *a, **k: object())
    monkeypatch.setattr(chatmod, "DeidentifyingLLM", lambda *a, **k: object())
    monkeypatch.setattr(chatmod, "HybridRetriever", lambda *a, **k: object())
    monkeypatch.setattr(chatmod, "Orchestrator", lambda **k: object())

    chatmod.build_orchestrator(embedder="hashing", llm="echo")
    assert seen["reset"] is False        # append, not reset — preserves event-report narratives
