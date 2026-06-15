"""Outbound calling — place a call to the single configured number, with the §6.1 retry policy.

POC: one hard-configured destination, no routing/escalation tree (O3). `place_with_retries`
encodes the failure policy: invalid number ⇒ fail fast (no retry); no-answer/busy ⇒ retry up to
`OUTBOUND_MAX_RETRIES` with `OUTBOUND_RETRY_DELAY_S` between attempts, then give up. The delay is
injectable so tests don't actually sleep.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from enum import Enum

from common.config import DEFAULT, Config
from common.notify import is_valid_number  # shared with the text-notify channel

__all__ = ["CallOutcome", "Caller", "SimulatedCaller", "LiveKitCaller", "get_caller",
           "is_valid_number", "place_with_retries", "parse_ack"]


class CallOutcome(str, Enum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    INVALID = "invalid"
    DROPPED = "dropped"  # mid-call (not a ring outcome)


class Caller(ABC):
    @abstractmethod
    def place_call(self, number: str) -> CallOutcome: ...


class SimulatedCaller(Caller):
    """Returns scripted ring outcomes (one per attempt); defaults to ANSWERED when exhausted."""

    def __init__(self, outcomes: list[CallOutcome] | None = None) -> None:
        self._outcomes = list(outcomes or [CallOutcome.ANSWERED])
        self.calls = 0

    def place_call(self, number: str) -> CallOutcome:
        self.calls += 1
        return self._outcomes.pop(0) if self._outcomes else CallOutcome.ANSWERED


class LiveKitCaller(Caller):
    """Real outbound via LiveKit (Cloud or self-hosted) SIP — dials the number into a room.

    The room is then driven by the LiveKit agent (voice/livekit_agent.py) for real audio. Returns
    a coarse `CallOutcome`: ANSWERED if the SIP participant was created (call picked up when
    `wait_until_answered`), INVALID if LiveKit isn't configured, else NO_ANSWER on a SIP error.
    """

    def __init__(self, config: Config = DEFAULT, room: str | None = None) -> None:
        self.config = config
        self.room = room or config.livekit_sip_room

    def place_call(self, number: str) -> CallOutcome:  # pragma: no cover - needs LiveKit + SDK
        from .livekit_cloud import LiveKitClient, is_configured  # noqa: PLC0415

        if not is_configured(self.config):
            return CallOutcome.INVALID
        try:
            LiveKitClient(self.config).create_outbound_sip_call(room=self.room, number=number)
            return CallOutcome.ANSWERED
        except Exception:  # noqa: BLE001 - SIP/trunk error -> treat as no-answer (retry policy applies)
            return CallOutcome.NO_ANSWER


def get_caller(name: str = "simulated", config: Config = DEFAULT, **kwargs) -> Caller:
    """Build a caller: 'simulated' (default, offline) or 'livekit' (real SIP via LiveKit Cloud)."""
    if name == "livekit":
        return LiveKitCaller(config, **kwargs)
    return SimulatedCaller(**kwargs)


def place_with_retries(
    caller: Caller, number: str, config: Config = DEFAULT, *, sleep_fn=time.sleep
) -> tuple[CallOutcome, int]:
    """Place the call with the §6.1 retry policy. Returns (final_outcome, attempts)."""
    if not is_valid_number(number):
        return CallOutcome.INVALID, 0  # fail fast, no retry

    attempts = 0
    for attempt in range(config.outbound_max_retries + 1):
        attempts += 1
        outcome = caller.place_call(number)
        if outcome == CallOutcome.ANSWERED:
            return outcome, attempts
        if outcome == CallOutcome.INVALID:
            return outcome, attempts
        # NO_ANSWER / BUSY -> retry after the delay (unless this was the last attempt)
        if attempt < config.outbound_max_retries:
            sleep_fn(config.outbound_retry_delay_s)
    return CallOutcome.NO_ANSWER, attempts  # exhausted -> notify_failed


_ACK_YES = {"yes", "yeah", "yep", "acknowledge", "acknowledged", "confirm", "confirmed",
            "affirmative", "copy", "roger", "correct"}
_ACK_NO = {"no", "nope", "negative", "deny", "denied", "incorrect"}


def parse_ack(text: str) -> str:
    """Classify a verbal acknowledgment as 'yes' | 'no' | 'unclear' (G4)."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & _ACK_YES:
        return "yes"
    if tokens & _ACK_NO:
        return "no"
    return "unclear"
