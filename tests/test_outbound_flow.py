"""Phase 7 outbound flow: decision gate, call, spoken report, follow-ups, acknowledgment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from common.config import DEFAULT
from common.deid import RegexDeidentifier
from common.interfaces import ECGModel
from common.providers import DeidentifyingLLM, EchoLLM
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from orchestrator.orchestrator import Orchestrator
from orchestrator.outbound_flow import run_outbound, should_call
from voice.outbound import CallOutcome, SimulatedCaller

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_DOCS = Path(__file__).resolve().parents[1] / "docs"

_CFG = replace(
    DEFAULT, outbound_enabled=True, outbound_call_number="+15551234567",
    outbound_min_criticality="High", outbound_max_retries=2, outbound_retry_delay_s=30,
)


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


def _afib_event():
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0
    return process_window(w, _Afib(), MewsVitalsAnalysis())


def _no_sleep(_):
    pass


# --- decision gate (no infra) ---


def test_should_call_gate():
    ev = _afib_event()
    assert should_call(ev, _CFG)[0]  # AFib is High >= High
    assert not should_call(ev, replace(_CFG, outbound_enabled=False))[0]
    assert not should_call(ev, replace(_CFG, outbound_min_criticality="Critical"))[0]


def test_false_positive_never_calls():
    w = next(read_hdf5_file(_FIXTURE))

    class _Normal(ECGModel):
        def predict(self, window):
            return "NORMAL_SINUS", 0.99

    fp = process_window(w, _Normal(), MewsVitalsAnalysis())
    assert fp.is_false_positive
    assert not should_call(fp, _CFG)[0]


# --- full loop over live backends ---


@pytest.fixture()
def env(tmp_path):
    from kb.graph.driver import GraphDriver
    from kb.graph.ingest import ingest_patient_record
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
    ingest_patient_record(d, {
        "patient_id": "PT1155", "gender": "M", "age": 73,
        "comorbidities": ["atrial fibrillation", "hypertension"], "prior_diagnoses": ["none"],
        "symptoms": ["palpitations"], "surgeries": ["none"], "current_medications": ["beta-blocker"],
    }, bed=("Unit1", "Unit1-Bed05"))

    vector = VectorRetriever.build(store=QdrantStore.in_memory(), embedder_name="hashing")
    vector.index_dir(_DOCS)
    orch = Orchestrator(
        working=WorkingMemory(rc), hybrid=HybridRetriever(vector, d),
        episodic=EpisodicMemory.in_memory(),
        llm=DeidentifyingLLM(EchoLLM(), RegexDeidentifier()), driver=d,
    )
    yield d, vector, orch
    d.reset_all()
    d.close()


def _persist(env):
    from orchestrator.event_flow import process_device_event

    driver, vector, _ = env
    ev = _afib_event()
    process_device_event(ev, driver, vector, bed=("Unit1", "Unit1-Bed05"))
    return ev


def _status(driver, uuid):
    return driver.run_read("MATCH (e:MonitoredEvent {id:$u}) RETURN e.status AS s", u=uuid)[0]["s"]


@pytest.mark.infra
def test_answered_call_with_followup_and_ack(env):
    driver, _, orch = env
    ev = _persist(env)
    result = run_outbound(
        ev, driver=driver, orchestrator=orch, caller=SimulatedCaller([CallOutcome.ANSWERED]),
        utterances=["what is the rate control for atrial fibrillation", "yes I acknowledge", "yes"],
        config=_CFG, bed="Unit1-Bed05", sleep_fn=_no_sleep,
    )
    assert result.called and result.outcome == "answered"
    assert "atrial fibrillation" in result.spoken_report.lower()
    assert result.answers and "grounded answer" in result.answers[0]  # grounded follow-up
    assert result.acknowledged and result.status == "acknowledged"
    assert _status(driver, ev.window.event_id) == "acknowledged"  # persisted


@pytest.mark.infra
def test_no_answer_marks_notify_failed(env):
    driver, _, orch = env
    ev = _persist(env)
    result = run_outbound(
        ev, driver=driver, orchestrator=orch,
        caller=SimulatedCaller([CallOutcome.NO_ANSWER] * 3),
        utterances=[], config=_CFG, sleep_fn=_no_sleep,
    )
    assert result.outcome == "no_answer" and result.status == "notify_failed"
    assert _status(driver, ev.window.event_id) == "notify_failed"


@pytest.mark.infra
def test_dropped_call_mid_alert_stays_reported(env):
    driver, _, orch = env
    ev = _persist(env)
    # caller asks one question then the line drops before acknowledging
    result = run_outbound(
        ev, driver=driver, orchestrator=orch, caller=SimulatedCaller([CallOutcome.ANSWERED]),
        utterances=["what were the vitals", "yes I acknowledge", "yes"],
        config=_CFG, bed="Unit1-Bed05", sleep_fn=_no_sleep, drop_after=1,
    )
    assert result.dropped and not result.acknowledged
    assert result.status == "reported"  # unacknowledged -> eligible for retry
    assert _status(driver, ev.window.event_id) == "reported"


@pytest.mark.infra
def test_unclear_ack_left_reported(env):
    driver, _, orch = env
    ev = _persist(env)
    result = run_outbound(
        ev, driver=driver, orchestrator=orch, caller=SimulatedCaller([CallOutcome.ANSWERED]),
        utterances=["yes acknowledge", "um not sure", "still unsure"],
        config=_CFG, bed="Unit1-Bed05", sleep_fn=_no_sleep,
    )
    assert not result.acknowledged and result.status == "reported"
    assert _status(driver, ev.window.event_id) == "reported"


@pytest.mark.infra
def test_invalid_number_fails_fast(env):
    driver, _, orch = env
    ev = _persist(env)
    result = run_outbound(
        ev, driver=driver, orchestrator=orch, caller=SimulatedCaller([CallOutcome.ANSWERED]),
        utterances=[], config=replace(_CFG, outbound_call_number="bogus"), sleep_fn=_no_sleep,
    )
    assert result.outcome == "invalid" and result.status == "notify_failed"


# --- text-notification channel (alternative to the voice call) ---


@pytest.mark.infra
def test_text_notify_delivered_with_followup_and_ack(env):
    from common.notify import SimulatedSmsNotifier
    from orchestrator.outbound_flow import run_text_notify

    driver, _, orch = env
    ev = _persist(env)
    sms = SimulatedSmsNotifier()
    result = run_text_notify(
        ev, driver=driver, orchestrator=orch, notifier=sms, to="+15551234567",
        utterances=["what is the rate control for atrial fibrillation", "yes I acknowledge", "yes"],
        config=_CFG, bed="Unit1-Bed05",
    )
    assert result.channel == "text" and result.outcome == "delivered"
    # the alert + the follow-up answer + confirm-back were all sent as messages
    messages = [m for _, m in sms.sent]
    assert any("atrial fibrillation" in m.lower() for m in messages)  # alert
    assert any("grounded answer" in m for m in messages)  # follow-up answered by text
    assert result.acknowledged and _status(driver, ev.window.event_id) == "acknowledged"


@pytest.mark.infra
def test_text_notify_delivery_failure(env):
    from common.notify import SimulatedSmsNotifier
    from orchestrator.outbound_flow import run_text_notify

    driver, _, orch = env
    ev = _persist(env)
    result = run_text_notify(
        ev, driver=driver, orchestrator=orch, notifier=SimulatedSmsNotifier(deliver=False),
        to="+15551234567", utterances=[], config=_CFG, bed="Unit1-Bed05",
    )
    assert result.outcome == "failed" and result.status == "notify_failed"
    assert _status(driver, ev.window.event_id) == "notify_failed"
