"""Phase 6 inbound voice: PIN gate before PHI, then grounded spoken answers."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from common.audit import AuditLog
from common.config import DEFAULT
from common.deid import RegexDeidentifier
from common.providers import DeidentifyingLLM, EchoLLM
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from orchestrator.orchestrator import Orchestrator
from voice.adapters import StubSTT, StubTTS
from voice.auth import PinAuthGate, parse_pin
from voice.handlers import OrchestratorHandler
from voice.session import VoiceSession

_DOCS = Path(__file__).resolve().parents[1] / "docs"

P1 = {"patient_id": "PT5001", "gender": "M", "age": 75,
      "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
      "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"]}
P2 = {"patient_id": "PT5002", "gender": "F", "age": 70,
      "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
      "symptoms": ["dyspnea"], "surgeries": ["none"], "current_medications": ["ace-inhibitor"]}


# --- PIN parsing (no infra) ---


def test_parse_pin_spoken_and_dtmf():
    assert parse_pin("one two three four") == "1234"
    assert parse_pin("1234") == "1234"
    assert parse_pin("my pin is 1 2 3 4") == "1234"
    assert parse_pin("hello there") == ""


def test_gate_verify():
    gate = PinAuthGate()  # default pin 1234
    assert gate.verify("one two three four")
    assert not gate.verify("nine nine nine nine")


# --- handler over live backends ---


@pytest.fixture()
def handler(tmp_path):
    from kb.graph.driver import GraphDriver
    from kb.graph.ingest import derive_comorbidity, ingest_patient_record
    from kb.graph.schema import migrate
    from memory.working import WorkingMemory

    try:
        d = GraphDriver.from_config(DEFAULT)
        d.run_read("RETURN 1 AS ok")
        import redis
        rc = redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2)
        rc.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend unreachable: {exc}")
    d.reset_all()
    migrate(d)
    ingest_patient_record(d, P1)
    ingest_patient_record(d, P2)
    derive_comorbidity(d)

    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    vector.index_dir(_DOCS)
    orch = Orchestrator(
        working=WorkingMemory(rc),
        hybrid=HybridRetriever(vector, d),
        episodic=EpisodicMemory.in_memory(),
        llm=DeidentifyingLLM(EchoLLM(), RegexDeidentifier()),
        driver=d,
    )
    audit = AuditLog(tmp_path / "audit.jsonl")
    h = OrchestratorHandler(orch, WorkingMemory(rc), audit=audit)
    h._audit_path = tmp_path / "audit.jsonl"
    yield h
    d.reset_all()
    d.close()


def _sid():
    return f"call-{uuid.uuid4().hex[:8]}"


def test_phi_refused_before_authentication(handler):
    sid = _sid()
    reply = handler.respond("what is the rate control for atrial fibrillation", session_id=sid)
    assert "PIN" in reply  # prompted to authenticate
    assert "grounded answer" not in reply  # orchestrator NOT invoked -> no PHI voiced
    handler.working.clear(sid)


def test_wrong_pin_then_lockout(handler):
    sid = _sid()
    for _ in range(3):
        reply = handler.respond("nine nine nine nine", session_id=sid)
    assert "Ending the call" in reply  # locked after max attempts
    handler.working.clear(sid)


def test_authenticated_caller_gets_grounded_answer(handler):
    sid = _sid()
    ok = handler.respond("one two three four", session_id=sid)
    assert "authenticated" in ok.lower()
    reply = handler.respond("what is the first-line management for atrial fibrillation", session_id=sid)
    assert "grounded answer" in reply  # orchestrator answered (grounded)
    handler.working.clear(sid)


def test_auth_and_query_audited(handler):
    sid = _sid()
    handler.respond("one two three four", session_id=sid)
    handler.respond("what is the first-line management for atrial fibrillation", session_id=sid)
    records = AuditLog(handler._audit_path).read_all()
    actions = {r["action"] for r in records}
    assert "inbound_auth" in actions and "phi_voice_query" in actions
    assert any(r["outcome"] == "success" for r in records if r["action"] == "inbound_auth")
    handler.working.clear(sid)


def test_spoken_grounded_answer_via_voice_session(handler):
    sid = _sid()
    session = VoiceSession(StubSTT(), StubTTS(), handler, sid)
    # turn 1: authenticate by "speaking" the PIN
    session.handle_turn(StubTTS().synthesize("one two three four"))
    # turn 2: a grounded question, spoken; measure latency under real retrieval
    result = session.handle_turn(
        StubTTS().synthesize("first line management for atrial fibrillation")
    )
    assert "grounded answer" in result.spoken
    assert StubSTT().transcribe(result.audio) == result.spoken  # heard back correctly
    assert result.metrics.full_turn_ms >= 0.0
    handler.working.clear(sid)
