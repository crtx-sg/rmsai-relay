"""Live end-to-end call 2026-11-06 (h1 + h2): fixes for what the transcripts showed."""

from __future__ import annotations

import time

from common.schemas import ChatTurn
from tests.test_episodic_gate import _orch, _SpyEpisodic

_ALERT = "Alert for patient PT970326 on bed 21. Detected sinus bradycardia."
_DECLINE = "I don't have information on that in the knowledge base."


def _seeded():
    orch = _orch(_SpyEpisodic(), episodic_recall=False)
    turns = orch.working.get_or_create("s").turns
    turns += [ChatTurn(role="assistant", text=_ALERT, timestamp=time.time(), mode="alert"),
              ChatTurn(role="assistant", text=_DECLINE, timestamp=time.time())]
    return orch


def test_repeat_skips_a_decline_and_revoices_the_alert():
    orch = _seeded()
    assert orch.handle_turn("s", "Repeat", spoken=True).answer == _ALERT
    assert orch.handle_turn("s", "repeat", spoken=True).answer == _ALERT  # not the repeat's echo


def test_pardon_on_a_call_is_a_repeat():
    for said in ("Eh?", "sorry?", "Pardon me?", "Hello?", "what?"):
        r = _seeded().handle_turn("s", said, spoken=True)
        assert (r.mode, r.answer) == ("repeat", _ALERT), said
    assert _seeded().handle_turn("s", "Hello?").mode != "repeat"  # typed chat: unchanged


def test_natural_pickup_replies_start_the_alert():
    from voice.handlers import is_pickup_reply

    for said in ("Hello?", "Hello? Hello?", "Hi?", "Hello.", "hello", "Moshi moshi.", "Yes, tell me", "Okay, go ahead",
                 "Go ahead", "Yes I am listening", "Yeah, speaking", "Sure."):
        assert is_pickup_reply(said), said
    # voicemail words and long turns refuse
    for said in ("Hello, you have reached the on-call phone", "Yes, call me back later",
                 "Hi, it's me, leave a message", "No", "Tell me about the patient", "1234",
                 "Yes, what is this?"):
        assert not is_pickup_reply(said), said


def test_without_a_pin_a_bare_yes_after_the_alert_does_not_acknowledge(tmp_path):
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking, _handler

    driver_calls = []
    h, _ = _handler(tmp_path, _FakeWorking(), _FakeOrch(), require_pin=False)
    h.driver = None
    h.respond("Yes", session_id="room1")  # the pickup "yes": the alert plays
    reply = h.respond("Yes", session_id="room1", spoken=True)  # said again in the silence
    assert reply.startswith("To close the alert") and reply.endswith("say acknowledge.")
    assert not h.is_ended("room1") and driver_calls == []
    assert h.respond("yeah", session_id="room1", spoken=True) == ""  # backchannel stays silent
    assert h.respond("Acknowledged", session_id="room1", spoken=True).endswith("Goodbye.")


def test_with_a_pin_yes_still_acknowledges(tmp_path):
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking, _handler

    working = _FakeWorking()
    h, _ = _handler(tmp_path, working, _FakeOrch(), require_pin=True)
    working.set_authenticated("room1", patient_ref="PT1155")
    assert h.respond("Yes", session_id="room1", spoken=True).endswith("Goodbye.")


def test_a_cut_off_alert_is_replayed_on_the_next_turn(tmp_path):
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking, _handler

    orch = _FakeOrch()
    h, alert = _handler(tmp_path, _FakeWorking(), orch, require_pin=False)
    played = h.respond("Yes", session_id="room1")
    h.note_interrupted("room1", "I don't have information on that.")  # not the alert: ignored
    h.note_interrupted("room1", played)  # room talk cut the alert off
    replay = h.respond("Orientation module", session_id="room1", spoken=True)
    assert replay == "Here is the alert again. " + alert.spoken_alert and orch.calls == []
    assert h.respond("what were the vitals", session_id="room1") == "answer:what were the vitals"
    h.note_interrupted("room1", replay)  # cut again, but the clinician acknowledges
    assert h.respond("Acknowledge", session_id="room1", spoken=True).endswith("Goodbye.")
    h2, _ = _handler(tmp_path, _FakeWorking(), _FakeOrch(), require_pin=False)
    h2.note_interrupted("r2", h2.respond("Yes", session_id="r2"))
    assert "recorded your acknowledgment" in h2.respond("I acknowledge. What is the BP?",
                                                        session_id="r2")


def test_an_ack_with_a_question_records_the_ack_and_answers(tmp_path):
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking, _handler

    orch = _FakeOrch()
    h, _ = _handler(tmp_path, _FakeWorking(), orch, require_pin=False)
    h.respond("Yes", session_id="room1")
    said = "I acknowledge. Um, can you qu- tell me what the patient's blood pressure is?"
    reply = h.respond(said, session_id="room1")
    assert reply.startswith("answer:Um, can you qu- tell me what the patient")
    assert reply.endswith("I've recorded your acknowledgment of the alert for patient PT1155 on bed ICU/1.")
    assert not h.is_ended("room1")  # the call stays up: the question was answered, not cut off
    audit = (tmp_path / "a.jsonl").read_text()
    assert '"action":"acknowledgment"' in audit
    # a question about acknowledging is still only a question
    h2, _ = _handler(tmp_path, _FakeWorking(), _FakeOrch(), require_pin=False)
    h2.respond("Yes", session_id="r2")
    assert "recorded your acknowledgment" not in h2.respond("Is it acknowledged? What is the BP?",
                                                             session_id="r2")


def test_any_other_patients_with_a_rhythm_is_the_graph_list_not_the_llm():
    from kb.graph.lookup import match_intent

    for q in ("Could you tell me if there are any other patients in the ward with atrial fibrillation?",
              "Does any other patient have atrial fibrillation?", "any patients in the ward with VT"):
        assert match_intent(q, now=time.time(), patient_ref="PT1", event_ref="E1")[0] == \
            "patients_with_event_type", q
    assert match_intent("Does the patient have atrial fibrillation?", now=time.time(),
                        patient_ref="PT1", event_ref="E1") is None  # this patient: unchanged


def test_pre_alert_speech_is_logged_without_digits_or_names_when_there_is_no_pin():
    from voice.livekit_agent import pre_auth_heard

    assert pre_auth_heard("one two 1234", require_pin=True) == "(12 chars, before the PIN)"
    logged = pre_auth_heard("Hello, this is 1234", require_pin=False, deid_backend="regex")
    assert "1234" not in logged and "####" in logged and logged.startswith("'Hello")


def _hist_row(ts=10_000.0):
    # heart rate every 10 minutes for the hour before the event; BP every hour for 3 hours
    return {"patient": "PT950445", "bed": "Unit2/Unit2-Bed16", "ts": ts,
            "hr": [100, 110, 120, 130, 140, 150, 152], "hr_ts": [ts - 3600 + 600 * i for i in range(7)],
            "sbp": [130, 128, 124], "sbp_ts": [ts - 7200, ts - 3600, ts],
            "dbp": [70, 68, 66], "dbp_ts": [ts - 7200, ts - 3600, ts]}


def test_a_vital_two_hours_ago_is_the_reading_nearest_then():
    from orchestrator.orchestrator import _answer_vital_past

    bp = _answer_vital_past(_hist_row(), "bp", 7200)
    assert "systolic blood pressure 130" in bp and "2 hours before the event" in bp
    hr = _answer_vital_past(_hist_row(), "hr", 7200)  # HR is only kept for the last hour
    assert "readings go back only 1 hour" in hr and "the earliest was 100" in hr
    assert "heart rate 130" in _answer_vital_past(_hist_row(), "hr", 1800)


def test_has_the_bp_been_stable_gives_its_course():
    from orchestrator.orchestrator import _answer_vital_past

    out = _answer_vital_past(_hist_row(), "bp", None)
    assert "systolic blood pressure went from 130 to 124 over the 2 hours" in out
    assert "lowest 124, highest 130, 3 readings" in out
    # an old event without reading times still answers from the values
    old = {k: v for k, v in _hist_row().items() if not k.endswith("_ts")}
    assert "their times were not recorded" in _answer_vital_past(old, "hr", 7200)


def test_past_and_course_questions_route_to_the_stored_readings():
    from kb.graph.lookup import match_intent

    def route(q):
        return match_intent(q, now=time.time(), patient_ref="PT1", event_ref="E1")

    assert route("What was the heart rate two hours ago?")[1] == {
        "event_uuid": "E1", "ago_s": 7200, "focus": "hr"}
    assert route("Has the blood pressure been stable?")[1]["course"] is True
    assert route("What was the blood pressure some time ago?")[1]["focus"] == "bp"
    assert route("What was the heart rate?")[0] == "vitals_at_selected_event"  # unchanged
    assert route("Is the patient deteriorating?")[0] == "event_summary"  # unchanged


def test_ward_questions_route_to_the_ward_not_this_patient():
    from kb.graph.lookup import match_intent

    def route(q):
        return match_intent(q, now=time.time(), patient_ref="PT1", event_ref="E1")

    assert route("What were the other events of other patients in this unit?") == (
        "event_history", {"status": "any"})
    for q in ("How many patients are critical?", "How many patient the critical conditions?",
              "What is the count of number of patients who are critical in this unit?"):
        assert route(q) == ("patients_by_criticality", {"count": True}), q
    assert route("what are the conditions of this patient")[0] == "patient_clinical_history"
    assert route("How many patients have high blood pressure?") != (
        "patients_by_criticality", {"count": True})


def test_critical_count_counts_open_critical_and_high_alerts():
    from orchestrator.orchestrator import _answer_criticality_count

    rows = [{"bed": "Unit1/Unit1-Bed02", "open_criticality": "Critical", "open_alerts": 2},
            {"bed": "Unit1/Unit1-Bed19", "open_criticality": "High", "open_alerts": 1},
            {"bed": "Unit1/Unit1-Bed13", "open_criticality": "High", "open_alerts": 1},
            {"bed": "Unit1/Unit1-Bed05", "open_criticality": None, "open_alerts": 0}]
    assert _answer_criticality_count(rows) == (
        "Of 4 patients, 3 have open critical or high alerts: 1 critical (bed 2), "
        "2 high (bed 19, bed 13).")
    assert "none has" in _answer_criticality_count(rows[3:])


def test_a_dial_that_never_rang_does_not_use_up_the_ring_attempts():
    from dataclasses import replace

    from common.config import DEFAULT
    from voice.outbound import CallOutcome, SimulatedCaller, place_with_retries

    cfg = replace(DEFAULT, outbound_max_retries=2, outbound_retry_delay_s=30)
    outage_then_rings = [CallOutcome.DIAL_FAILED] * 5 + [CallOutcome.NO_ANSWER] * 2 + [
        CallOutcome.ANSWERED]
    caller = SimulatedCaller(outage_then_rings)
    assert place_with_retries(caller, "+15551234567", cfg, sleep_fn=lambda _s: None) == (
        CallOutcome.ANSWERED, 8)  # 5 failed dials, then still 3 ring attempts
    waits = []
    caller = SimulatedCaller([CallOutcome.DIAL_FAILED] * 20)
    assert place_with_retries(caller, "+15551234567", cfg, sleep_fn=waits.append) == (
        CallOutcome.DIAL_FAILED, 10)  # gives up after ~5 minutes of outage
    assert waits == [30] * 9


def test_only_auth_and_network_errors_count_as_never_rang():
    from voice.outbound import never_rang

    class _Http(Exception):
        def __init__(self, status):
            self.status = status

    assert never_rang(_Http(401)) and never_rang(ConnectionError("dns")) and never_rang(OSError())
    assert not never_rang(_Http(408)) and not never_rang(RuntimeError("sip 486 busy"))
    assert not never_rang(TimeoutError())  # timed out waiting for the pickup: it rang


def test_a_call_that_never_rang_texts_could_not_call_not_missed_call(tmp_path):
    from tests.test_sms_fallback import _RealSms, _run
    from voice.outbound import CallOutcome

    sms = _RealSms()
    result, _status, _ = _run([CallOutcome.DIAL_FAILED] * 10, sms, tmp_path)
    assert result.outcome == "dial_failed" and result.fallback == "sms_delivered"
    (_to, text), = sms.sent
    assert text.startswith("RMS relay could not call you.")


def test_similar_events_are_the_called_events_rhythm():
    from kb.graph.lookup import match_intent
    from kb.graph.templates import resolve_like_event

    def route(q):
        return match_intent(q, now=time.time(), patient_ref="PT1", event_ref="E1")

    for q in ("Are there any similar events for patients in this unit?",
              "any other patients with the same rhythm?", "events like this on the ward",
              "any similar unacknowledged alerts in the last 6 hours"):
        assert route(q)[1].get("like_event") == "E1", q
    assert route("similar events for this patient")[1]["patient_id"] == "PT1"
    assert route("What rhythm was detected?")[0] == "event_summary"  # unchanged

    class _D:
        def run_read(self, _cypher, **kw):
            return [{"t": "SINUS_TACHYCARDIA"}] if kw["uuid"] == "E1" else []

    assert resolve_like_event(_D(), {"status": "any", "like_event": "E1"}) == {
        "status": "any", "event_type": "SINUS_TACHYCARDIA"}
    assert resolve_like_event(_D(), {"status": "any", "like_event": "E9"}) == {"status": "any"}
