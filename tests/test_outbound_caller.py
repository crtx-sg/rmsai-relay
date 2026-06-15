"""Outbound caller retry policy (§6.1) + acknowledgment parsing (G4)."""

from __future__ import annotations

from dataclasses import replace

from common.config import DEFAULT
from voice.outbound import (
    CallOutcome,
    SimulatedCaller,
    is_valid_number,
    parse_ack,
    place_with_retries,
)

_CFG = replace(DEFAULT, outbound_max_retries=2, outbound_retry_delay_s=30)


def _no_sleep(_seconds):  # never actually wait in tests
    pass


def test_valid_number():
    assert is_valid_number("+15551234567")
    assert is_valid_number("5551234567")
    assert not is_valid_number("")
    assert not is_valid_number("not-a-number")


def test_invalid_number_fails_fast_no_retry():
    caller = SimulatedCaller([CallOutcome.ANSWERED])
    outcome, attempts = place_with_retries(caller, "bogus", _CFG, sleep_fn=_no_sleep)
    assert outcome == CallOutcome.INVALID
    assert caller.calls == 0  # never dialed


def test_answered_on_first_attempt():
    caller = SimulatedCaller([CallOutcome.ANSWERED])
    outcome, attempts = place_with_retries(caller, "+15551234567", _CFG, sleep_fn=_no_sleep)
    assert outcome == CallOutcome.ANSWERED and attempts == 1


def test_no_answer_retries_then_gives_up():
    # max_retries=2 -> 3 total attempts, all no-answer -> notify_failed
    caller = SimulatedCaller([CallOutcome.NO_ANSWER] * 5)
    delays = []
    outcome, attempts = place_with_retries(
        caller, "+15551234567", _CFG, sleep_fn=lambda s: delays.append(s)
    )
    assert outcome == CallOutcome.NO_ANSWER
    assert attempts == 3  # 1 + 2 retries
    assert delays == [30, 30]  # waited between attempts, not after the last


def test_busy_then_answered():
    caller = SimulatedCaller([CallOutcome.BUSY, CallOutcome.ANSWERED])
    outcome, attempts = place_with_retries(caller, "+15551234567", _CFG, sleep_fn=_no_sleep)
    assert outcome == CallOutcome.ANSWERED and attempts == 2


def test_parse_ack():
    assert parse_ack("yes I acknowledge") == "yes"
    assert parse_ack("acknowledged") == "yes"
    assert parse_ack("no") == "no"
    assert parse_ack("what were the vitals") == "unclear"
