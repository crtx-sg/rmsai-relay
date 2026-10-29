"""Live call 2026-10-29: a pickup that hung up in 4 s with no PIN was logged "answered", so it
was not rung again and no SMS went out. Answered now means the PIN step was reached."""
from __future__ import annotations

from dataclasses import replace

from common.config import DEFAULT
from orchestrator.bus_consumer import _PinGatedCaller
from orchestrator.call_queue import CallQueue
from tests.test_call_queue import _FakeRedis
from voice.outbound import CallOutcome, SimulatedCaller, place_with_retries


class _Store:
    def __init__(self):
        self.redis = _FakeRedis()


def _caller(store, ended):
    return _PinGatedCaller(SimulatedCaller([CallOutcome.ANSWERED] * 5), "room", "e",
                           lambda room: ended.append(room) or 4.0, store)


def test_pickup_without_a_pin_is_missed_and_retried_to_the_last_attempt():
    store, ended = _Store(), []
    cfg = replace(DEFAULT, outbound_max_retries=2)
    outcome, attempts = place_with_retries(_caller(store, ended), "+10000000000", cfg,
                                           sleep_fn=lambda s: None)
    assert (outcome, attempts, len(ended)) == (CallOutcome.NO_ANSWER, 3, 3)  # -> SMS fallback


def test_right_pin_or_lockout_is_answered():
    for mark in ("mark_answered", "mark_locked"):
        store = _Store()
        getattr(CallQueue(store.redis), mark)("e")
        assert _caller(store, []).place_call("+10000000000") == CallOutcome.ANSWERED


def test_unanswered_ring_does_not_wait_for_a_call_end():
    store, ended = _Store(), []
    c = _PinGatedCaller(SimulatedCaller([CallOutcome.NO_ANSWER]), "room", "e", ended.append, store)
    assert c.place_call("+10000000000") == CallOutcome.NO_ANSWER and ended == []


def test_lockout_tells_the_queue(tmp_path):
    from common.audit import AuditLog  # noqa: PLC0415
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking  # noqa: PLC0415
    from voice.handlers import OutboundHandler  # noqa: PLC0415
    from voice.outbound_alert import OutboundAlert  # noqa: PLC0415

    locked = []
    h = OutboundHandler(_FakeOrch(), _FakeWorking(), OutboundAlert("r", "PT1", "e", "x", "ICU/1"),
                        on_locked=locked.append, audit=AuditLog(tmp_path / "a.jsonl"))
    for _ in range(h.max_attempts):
        h.respond("nine nine nine nine", session_id="r")
    assert locked == ["e"] and h.is_locked("r")


# --- SMS test 2026-10-29: a reconnect while ringing rang a second round after the SMS; keyed PIN
# digits were ignored ------------------------------------------------------------------------------

class _Room:
    name = "rmsai-outbound-e"

    def __init__(self):
        self.handlers = {}

    def on(self, event, fn):
        self.handlers[event] = fn


def test_reconnect_before_the_pin_hangs_up_without_a_redial_flag():
    import asyncio  # noqa: PLC0415

    from voice.livekit_agent import end_call_on_reconnect  # noqa: PLC0415

    room, marked, deleted = _Room(), [], []

    async def _delete():
        deleted.append(True)

    async def _run():
        end_call_on_reconnect(room, "e", mark_dropped=marked.append, delete_room=_delete,
                              answered=lambda: False)
        room.handlers["reconnected"]()
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert marked == [] and deleted == [True]


def test_unanswered_round_is_not_redialled_even_if_flagged(monkeypatch):
    from inference.serialize import event_to_dict  # noqa: PLC0415
    from orchestrator import bus_consumer  # noqa: PLC0415
    from orchestrator.call_queue import CallJob  # noqa: PLC0415
    from orchestrator.outbound_flow import OutboundResult  # noqa: PLC0415
    from tests.test_bus_consumer import _Afib, _event  # noqa: PLC0415

    store, dials = _Store(), []
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: "reported")
    monkeypatch.setattr(bus_consumer, "_record_delivery", lambda *a, **k: None)

    def _place(*a, **k):
        dials.append(1)
        CallQueue(store.redis).mark_dropped("e")  # the worker flagged it during this round
        return OutboundResult(called=True, decision_reason="ok", outcome="no_answer", attempts=3,
                              status="reported", fallback="sms_delivered")

    monkeypatch.setattr(bus_consumer, "_place_live_call", _place)
    job = CallJob(event_id="e", patient_ref="PT1", event_type="ATRIAL_FIBRILLATION", bed="ICU/3",
                  criticality="High", reason="ok", payload=event_to_dict(_event(_Afib())))
    bus_consumer.run_queued_call(job, driver=object(), orchestrator=object(), utterances=[],
                                 caller_factory=lambda r: object(), alert_store=store)
    assert dials == [1]


def test_keypad_pin_is_one_turn():
    from voice.livekit_agent import collect_dtmf  # noqa: PLC0415

    class _D:
        def __init__(self, digit):
            self.digit = digit

    room, turns = _Room(), []
    collect_dtmf(room, turns.append)
    for d in "12*1234":
        room.handlers["sip_dtmf_received"](_D(d))
    for d in "99#":
        room.handlers["sip_dtmf_received"](_D(d))
    assert turns == ["1234", "99"]


def test_each_redial_dials_a_fresh_room():
    from voice.outbound import LiveKitCaller  # noqa: PLC0415

    class _Rec(LiveKitCaller):
        def place_call(self, number):
            dialled.append(self.room)
            return CallOutcome.NO_ANSWER

    dialled = []
    gated = _PinGatedCaller(_Rec(DEFAULT, room="rmsai-outbound-e"), "rmsai-outbound-e", "e", None,
                            _Store())
    rooms = iter(["rmsai-outbound-e-r1", "rmsai-outbound-e-r2"])

    def _next():
        gated.room = gated.caller.room = next(rooms)

    place_with_retries(gated, "+10000000000", replace(DEFAULT, outbound_max_retries=2),
                       sleep_fn=lambda s: None, before_retry=_next)
    assert dialled == ["rmsai-outbound-e", "rmsai-outbound-e-r1", "rmsai-outbound-e-r2"]
