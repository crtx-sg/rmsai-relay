"""Offline tests for wake-word detection (voice/wake.py) — pure function, no models."""

from __future__ import annotations

import pytest

from voice.wake import detect_wake_word


@pytest.mark.parametrize(
    "text,remainder",
    [
        ("hey vios what were the vitals", "what were the vitals"),
        ("Hey, Vios! Show the vitals.", "show the vitals"),
        ("hey vios", ""),                                  # wake word alone -> awake, no question
        ("um, hey vios, get the patient history", "get the patient history"),  # leading noise
        ("hey bios what is the mews", "what is the mews"),  # STT variant of "vios"
        ("hello vios status please", "status please"),     # greeting variant
        ("aveos what were the vitals", "what were the vitals"),  # merged single-token mishearing
        ("Avios, show the vitals.", "show the vitals"),    # another merged form
        ("a vios get the patient history", "get the patient history"),  # split lead-in
        # Real mishearings seen in live calls — greeting-anchored stripping handles any "vios" form:
        ("Hey, why us? What's the status of?", "whats the status of"),
        ("hey vyas, show the report", "show the report"),
        ("hi there, what are the vitals", "what are the vitals"),   # 2-token junk before question
    ],
)
def test_wake_word_matches_and_strips(text, remainder):
    matched, rem = detect_wake_word(text, "hey vios")
    assert matched is True
    assert rem == remainder


@pytest.mark.parametrize(
    "text",
    [
        "what were the vitals at the time of the event",  # no wake word (e.g. noise/hallucination)
        "it's been a lot of years",                        # the Whisper-on-silence hallucination
        "vios",                                            # brand word without the greeting
        "",
    ],
)
def test_no_wake_word(text):
    assert detect_wake_word(text, "hey vios") == (False, "")


def test_custom_wake_word():
    assert detect_wake_word("computer, what's the heart rate", "computer") == (
        True,
        "whats the heart rate",
    )
    assert detect_wake_word("hey vios what's up", "computer") == (False, "")
