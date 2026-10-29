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
