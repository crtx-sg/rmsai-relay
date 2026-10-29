"""Live calls 2026-10-28: audio died after a LiveKit reconnect; "give the last updated blood
pressure" was refused; Hindi/Tamil background talk was answered; "Are there any" was answered on
its own; "acknowledge this evening" was not an ack; "previous events" read the vitals; "trends that
are deteriorating" went to an ST-elevation query."""
from __future__ import annotations

import asyncio
import json

from inference.serialize import event_to_dict
from kb.graph.lookup import match_intent
from orchestrator import bus_consumer
from orchestrator.call_queue import CallJob, CallQueue
from orchestrator.guardrails import Guardrails
from orchestrator.orchestrator import _answer_event_summary
from tests.test_bus_consumer import _Afib, _event, patched  # noqa: F401
from tests.test_call_queue import _FakeRedis
from voice.handlers import is_filler
from voice.livekit_agent import end_call_on_reconnect, is_dangling
from voice.outbound import parse_ack


def test_give_the_vitals_is_a_question_not_an_order():
    g = Guardrails()
    assert g.check_input("Give the last updated blood pressure for this patient").allowed
    assert not g.check_input("give amiodarone 300 mg").allowed


def test_speech_without_latin_letters_is_noise():
    assert is_filler("अरे कहाँ का सर आ गए हैं सर आप?") and is_filler("அது இனி அந்தச் சம")
    assert not is_filler("what is the heart rate") and not is_filler("")


def test_are_there_any_waits_for_the_rest():
    assert is_dangling("Are there any")


def test_acknowledge_this_evening_is_an_ack():
    assert parse_ack("Acknowledge this evening") == "yes"


def test_previous_events_on_a_call_are_this_patients():
    assert match_intent("Previous events", now=0.0, patient_ref="PT1", event_ref="E1") == (
        "event_history", {"status": "any", "patient_id": "PT1"})


def test_deteriorating_trends_come_from_the_events_verdicts():
    q = "What are the trends that are are deteriorating for this e- for this patient"
    assert match_intent(q, now=0.0, patient_ref="PT1", event_ref="E1") == (
        "event_summary", {"event_uuid": "E1", "focus": "trends"})
    assert match_intent("which patients are deteriorating", now=0.0, patient_ref="PT1",
                        event_ref="E1") != ("event_summary", {"event_uuid": "E1", "focus": "trends"})
    why = json.dumps({"vitals": {"deteriorating": [
        {"vital": "HR", "direction": "rising", "change": 35.0, "unit": "bpm", "span_s": 3898,
         "latest": 194},
        {"vital": "SpO2", "direction": "falling", "change": -3.4, "unit": "%", "span_s": 1903,
         "latest": 84}]}})
    assert _answer_event_summary({"patient": "PT1", "why_json": why}, "trends") == (
        "Patient PT1: deteriorating: heart rate rising 35 beats per minute over 65 minutes, now 194;"
        " oxygen saturation falling 3.4 percent over 32 minutes, now 84.")
    none = json.dumps({"vitals": {"deteriorating": []}})
    assert _answer_event_summary({"patient": "PT1", "why_json": none}, "trends") == (
        "Patient PT1: no vital is deteriorating.")


class _Room:
    name = "rmsai-outbound-e"

    def __init__(self):
        self.handlers = {}

    def on(self, event, fn):
        self.handlers[event] = fn


def test_reconnect_hangs_up_and_marks_the_call_for_a_redial():
    room, marked, deleted = _Room(), [], []

    async def _delete():
        deleted.append(True)

    async def _run():
        end_call_on_reconnect(room, "e", mark_dropped=marked.append, delete_room=_delete)
        room.handlers["reconnecting"]()
        assert not deleted  # only once reconnected
        room.handlers["reconnected"]()
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert marked == ["e"] and deleted == [True]


class _Store:
    def __init__(self):
        self.redis = _FakeRedis()

    def put(self, alert):
        pass


def test_a_dropped_call_is_rung_again(patched, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bus_consumer, "_record_delivery", lambda *a, **k: None)
    store = _Store()
    job = CallJob(event_id="e", patient_ref="PT1", event_type="ATRIAL_FIBRILLATION", bed="ICU/3",
                  criticality="High", reason="ok", payload=event_to_dict(_event(_Afib())))
    real, calls = bus_consumer._place_live_call, []

    def _place(*a, **k):  # the worker flags the first call as dropped on a reconnect, mid-call
        calls.append(1)
        if len(calls) == 1:
            CallQueue(store.redis).mark_dropped("e")
        return real(*a, **k)

    monkeypatch.setattr(bus_consumer, "_place_live_call", _place)
    bus_consumer.run_queued_call(job, driver=object(), orchestrator=object(), utterances=[],
                                 caller_factory=lambda r: object(), alert_store=store)
    assert patched["voice"] == ["ICU/3", "ICU/3"]  # rung again once, then the flag is gone
