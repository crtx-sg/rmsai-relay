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

    for said in ("Hello?", "Hello? Hello?", "Hi?", "Moshi moshi.", "Yes, tell me", "Okay, go ahead",
                 "Go ahead", "Yes I am listening", "Yeah, speaking", "Sure."):
        assert is_pickup_reply(said), said
    # a flat "Hello." can be the first piece of a voicemail greeting; vetoes and long turns refuse
    for said in ("Hello.", "Hello, you have reached the on-call phone", "Yes, call me back later",
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
