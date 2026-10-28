"""Regressions found by the branch review of fix/audit-2026-10 (2026-10-14), one test per finding.
Phrasings are the reviewers' own; each produced the wrong result before the fix."""

from __future__ import annotations

import json

import pytest

from cli.consume import handle_entry, reclaim_pending
from common.audit import AuditLog
from kb.graph.lookup import is_patient_specific, match_intent, names_unresolved_bed
from kb.graph.spoken import normalize_spoken_query
from orchestrator.call_queue import CallQueue
from orchestrator.guardrails import check_input, ungrounded_identifiers
from orchestrator.orchestrator import _MEWS_SCORE, _answer_action_items, _answer_clinical_history
from voice.livekit_agent import can_carry, is_dangling
from voice.outbound import is_plain_no, parse_ack
from voice.outbound_alert import OutboundAlert

NOW = 1_000_000.0
S = {"now": NOW, "patient_ref": "PT1", "event_ref": "E1"}


# --- 1. acknowledgment commands only -------------------------------------------------------------

class _Orch:
    def handle_turn(self, sid, text, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(answer=f"answer:{text}", declined=False)


class _Working:
    def __init__(self):
        from types import SimpleNamespace
        self.s = SimpleNamespace(authenticated=False, patient_ref=None, event_ref=None)

    def get_or_create(self, sid):
        return self.s

    def append_turn(self, *_a, **_k):
        pass

    def set_authenticated(self, sid, *, patient_ref=None, event_ref=None):
        self.s.authenticated, self.s.patient_ref, self.s.event_ref = True, patient_ref, event_ref


def _call(tmp_path, monkeypatch, bed="Unit1/Unit1-Bed02"):
    import kb.graph.events as events
    from voice.handlers import OutboundHandler

    acked = []
    monkeypatch.setattr(events, "set_event_status", lambda d, u, s: acked.append(u))
    alert = OutboundAlert("r", "PT998224", "E-VF", "alert", bed, event_type="VENTRICULAR_FIBRILLATION")
    h = OutboundHandler(_Orch(), _Working(), alert, driver=object(),
                        audit=AuditLog(tmp_path / "a.jsonl"))
    h.respond("one two three four", session_id="r")
    return h, acked


@pytest.mark.parametrize("text", [
    "don't acknowledge bed two yet", "do not acknowledge PT998224", "is bed two acknowledged",
    "was bed two acknowledged?", "did anyone acknowledge PT998224", "should I acknowledge bed two?",
    "I can't acknowledge bed two right now"])
def test_negations_and_questions_never_acknowledge(tmp_path, monkeypatch, text):
    h, acked = _call(tmp_path, monkeypatch)
    h.respond(text, session_id="r")
    assert acked == []


def test_question_about_another_bed_is_answered_not_refused(tmp_path, monkeypatch):
    h, acked = _call(tmp_path, monkeypatch)
    assert h.respond("who acknowledged bed seven?", session_id="r").startswith("answer:")
    assert acked == []


def test_named_bed_is_refused_when_the_alert_has_no_bed(tmp_path, monkeypatch):
    h, acked = _call(tmp_path, monkeypatch, bed=None)
    assert "only acknowledge" in h.respond("acknowledge bed seven", session_id="r")
    assert acked == []


# --- 3. acknowledgment phrasing -------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "that's correct", "I've acknowledged it", "Yes sir", "yes ma'am", "Yes, it's acknowledged",
    "acknowledged, no further questions", "no problem, acknowledged", "Acknowledged. Thanks very much.",
    "acknowledged, bye", "Yes, I acknowledge receipt of the alert", "that's right"])
def test_common_acknowledgments(text):
    assert parse_ack(text) == "yes"


@pytest.mark.parametrize("text", ["not yet", "nope", "No, that is not correct", "don't acknowledge yet"])
def test_plain_refusals(text):
    assert parse_ack(text) == "no" and is_plain_no(text)


# --- 11. a "no" with a question is a question -----------------------------------------------------

@pytest.mark.parametrize("text", ["No, what is the heart rate", "No. Tell me the heart rate",
                                  "not yet, what's his potassium"])
def test_no_followed_by_a_question_is_answered(tmp_path, monkeypatch, text):
    assert not is_plain_no(text)
    h, _ = _call(tmp_path, monkeypatch)
    assert h.respond(text, session_id="r").startswith("answer:")


# --- 2, 13. held pieces --------------------------------------------------------------------------

@pytest.mark.parametrize("q", ["Can you tell me what the potassium was?",
                               "Do you know what the MEWS score is?", "Tell me what the heart rate was",
                               "Repeat what the alert was", "who is the patient this is about",
                               "tell me where he is", "I don't know what that is"])
def test_whole_questions_are_not_held(q):
    assert not is_dangling(q)


def test_no_carry_in_app_chat_before_the_pin_or_for_control_messages():
    assert not can_carry("what was the", authenticated=True, push_to_talk=True)
    assert not can_carry("/select-silent abc", authenticated=True, push_to_talk=False)
    assert not can_carry("1235", authenticated=False, push_to_talk=False)
    assert not can_carry("acknowledge this", authenticated=True, push_to_talk=False)
    assert can_carry("what was the", authenticated=True, push_to_talk=False)


# --- 5, 6. this-event rules stay out of other questions --------------------------------------------

@pytest.mark.parametrize("q, name", [
    ("what was the heart rate when the event was detected", "vitals_at_selected_event"),
    ("what was the MEWS when it was detected", "mews_at_selected_event"),
    ("what medications is he on for the arrhythmia", "patient_clinical_history"),
    ("does he have a history of arrhythmia", "patient_clinical_history"),
])
def test_specific_rules_win_over_this_event(q, name):
    assert match_intent(q, **S)[0] == name


@pytest.mark.parametrize("q", [
    "what is a false positive", "what is an artifact", "when is the next dose",
    "when was the patient admitted", "what time is it", "what is the accuracy of pulse oximetry",
    "what is the treatment for this rhythm",  # a protocol question (rewritten by the orchestrator)
    "what do we do for bradycardia", "what are the recommendations for AFib",
    "what action should I take for hyperkalemia", "what is the recommended dose now"])
def test_not_about_this_event(q):
    intent = match_intent(q, **S)
    assert intent is None or intent[0] not in ("event_summary", "action_items_for_event")


# --- 7. misheard bed words -------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["is that bad 2 days later?", "I bet two of them are false",
                                  "bid 2 times daily", "is this a bad one?"])
def test_ordinary_words_are_not_beds(text):
    assert "Unit1-Bed" not in normalize_spoken_query(text)


def test_misheard_bed_after_a_preposition_still_counts():
    assert "Unit1-Bed09" in normalize_spoken_query("last reported event on vid nine")
    assert "Unit1-Bed03" in normalize_spoken_query("vitals for bad number three")


# --- 8. document questions are not "patient-specific" -------------------------------------------------

@pytest.mark.parametrize("q", ["how do I treat her AFib", "what does the model do",
                               "when should I escalate the patient", "what does the alarm mean",
                               "what does the patient's MEWS mean", "what are the alarm limits now"])
def test_document_questions_reach_the_documents(q):
    assert not is_patient_specific(q)


def test_named_patient_is_always_patient_specific():
    assert is_patient_specific("what does PT4543 have")


@pytest.mark.parametrize("q", ["what is a bed bath", "the bed alarm keeps ringing, what do I do",
                               "how to clean the bed sensor"])
def test_bed_as_a_thing_is_not_a_missing_bed_number(q):
    assert not names_unresolved_bed(q)


# --- 10. the MEWS-meaning number -------------------------------------------------------------------

@pytest.mark.parametrize("q, score", [
    ("What does a MEWS score of 5 mean?", "5"), ("MEWS 3 means?", "3"),
    ("MEWS escalation in 30 minutes means?", None), ("what does MEWS mean for a 70 year old", None),
    ("what does it mean if the MEWS rises by 2 in 4 hours", None)])
def test_mews_meaning_reads_only_the_score(q, score):
    m = _MEWS_SCORE.search(q)
    assert (m.group(1) if m else None) == score


# --- 17. event-count windows ----------------------------------------------------------------------

@pytest.mark.parametrize("q, since, window", [
    ("how many events in the last 30 minutes", NOW - 1800, "in the last 30 minutes"),
    ("how many events in the last 2 days", NOW - 2 * 86400, "in the last 2 days"),
    ("how many events this week", NOW - 7 * 86400, "in the last 7 days")])
def test_event_count_windows(q, since, window):
    assert match_intent(q, now=NOW) == ("event_count_since", {"since": since, "window": window})


# --- 4, 18. consumer: retries and reclaim -------------------------------------------------------------

class _Redis:
    def __init__(self):
        self.acked, self.hash = [], {}

    def xack(self, s, g, mid):
        self.acked.append(mid)

    def hincrby(self, k, f, n):
        self.hash[f] = self.hash.get(f, 0) + n
        return self.hash[f]

    def hdel(self, k, f):
        self.hash.pop(f, None)

    def xautoclaim(self, *a, **k):
        return [b"0-0", [(b"1-0", {b"data": b"{}"}), (b"2-0", {})], []]


class _BrokenAudit:
    def write(self, **kw):
        raise OSError("read-only file system")


def test_an_unwritable_audit_log_never_crashes_the_loop():
    def down(p, m):
        raise ConnectionError("neo4j restarting")

    out = handle_entry(_Redis(), "s", "g", b"1-0", {b"data": b"{}"}, down, audit=_BrokenAudit())
    assert out == "retry"


def test_reclaim_returns_claimed_entries_with_data():
    assert reclaim_pending(_Redis(), "s", "g", "c1", min_idle_ms=1, count=10) == \
        [(b"1-0", {b"data": b"{}"})]


# --- 14. the call slot is compare-and-act ---------------------------------------------------------------

def test_a_lapsed_holder_cannot_renew_or_free_the_new_holders_slot():
    from tests.test_call_queue import _FakeRedis, _job

    r = _FakeRedis()
    a, b = CallQueue(r), CallQueue(r)
    a.push(_job("x", "Critical", 1))
    a.push(_job("y", "High", 2))
    assert a.claim().event_id == "x"
    r.delete(a.active_key)                 # A's lease lapsed (laptop asleep)
    assert b.claim().event_id == "x"       # B recovers and dials it
    assert a.keep_alive("x") is True       # same event id: harmless, still x's slot
    b.done("x")
    assert b.claim().event_id == "y"
    assert a.keep_alive("x") is False      # A wakes up: the slot is no longer its own
    a.done("x")
    assert r.get(a.active_key) == "y"      # ...and A cannot free B's call slot


# --- 9. SMS fallback survives a graph error -----------------------------------------------------------

def test_sms_fallback_is_sent_when_the_status_check_fails(tmp_path):
    from tests.test_sms_fallback import _CFG, _event, _RealSms
    from orchestrator.outbound_flow import run_outbound
    from voice.outbound import CallOutcome, SimulatedCaller

    class _Driver:
        def __init__(self):
            self.status = {}

        def run_write(self, _q, uuid, status, closed=()):
            self.status[uuid] = status

        def run_read(self, *_a, **_k):
            raise ConnectionError("neo4j restarting")

    sms = _RealSms()
    result = run_outbound(_event(), driver=_Driver(), orchestrator=None,
                          caller=SimulatedCaller([CallOutcome.NO_ANSWER] * 3), utterances=[],
                          config=_CFG, bed="Unit1-Bed05", audit=AuditLog(tmp_path / "a.jsonl"),
                          sleep_fn=lambda _s: None, fallback_notifier=sms)
    assert sms.sent and result.fallback == "sms_delivered"


# --- 19, 20. small items --------------------------------------------------------------------------------

def test_alert_from_a_newer_version_still_loads():
    raw = json.dumps({"session_id": "r", "patient_ref": "PT1", "event_id": "E", "spoken_alert": "x",
                      "bed": "b", "event_type": "VF", "added_later": 1})
    assert OutboundAlert.from_json(raw).event_type == "VF"


def test_citation_guard_ignores_clinical_text():
    assert ungrounded_identifiers("The patient is P2-positive.", "ctx") == []
    assert ungrounded_identifiers("The patient's ID is #P1.", "ctx") != []


def test_information_questions_opening_with_an_action_verb():
    assert check_input("shock energy for VF?").allowed
    assert not check_input("please give 1mg epinephrine").allowed


def test_renderers_handle_missing_fields():
    assert _answer_clinical_history([{"patient": "PT1", "age": 70}], "demographics") == \
        "Patient PT1 is 70 years old."
    assert _answer_clinical_history([{"patient": None}], "medications") == \
        "No medications are recorded for this patient."
    assert _answer_action_items([{"patient": "PT1", "action": None}]) == \
        "No action items are recorded for this event."


# --- docs review: replies to the alert pass the wake gate (as the README always said) -------------

@pytest.mark.parametrize("text, passes", [
    ("acknowledged", True), ("Acknowledge this event", True), ("yes", True), ("not yet", True),
    ("acknowledge bed two", True), ("don't acknowledge yet", True),
    ("is bed two acknowledged", False), ("what is the heart rate", False)])
def test_alert_replies_skip_the_wake_word(text, passes):
    from voice.outbound import is_ack_reply

    assert is_ack_reply(text) is passes
