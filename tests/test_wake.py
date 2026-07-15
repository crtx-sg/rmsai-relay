"""Offline tests for wake-word detection (voice/wake.py) — pure function, no models."""

from __future__ import annotations

import pytest

from voice.wake import detect_wake_word, gate_audio_turn


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


# --- gate_audio_turn: the pure decision behind on_user_turn_completed ---------------------------

def test_gate_answers_when_wake_word_leads():
    action, question, awake = gate_audio_turn(
        "hey vios what were the vitals", now=100.0, awake_window_s=30.0)
    assert action == "answer"
    assert question == "what were the vitals"  # wake phrase stripped
    assert awake == 130.0                       # window opened


def test_gate_bare_wake_word_opens_window_but_drops():
    action, question, awake = gate_audio_turn("hey vios", now=100.0, awake_window_s=30.0)
    assert action == "drop" and question is None
    assert awake == 130.0  # awake now, nothing to answer yet


def test_gate_answers_inside_open_window_without_wake_word():
    # A follow-up with no wake word, still inside the awake window, is answered and refreshes it.
    action, question, awake = gate_audio_turn(
        "and the blood pressure", awake_until=130.0, now=110.0, awake_window_s=30.0)
    assert action == "answer" and question is None  # answered as-is (no phrase to strip)
    assert awake == 140.0                            # window refreshed


def test_gate_drops_noise_when_window_closed():
    action, question, awake = gate_audio_turn(
        "it's been a lot of years", awake_until=100.0, now=200.0)
    assert action == "drop" and question is None
    assert awake == 100.0  # unchanged; stays asleep


def test_gate_bypassed_when_wake_not_required():
    # AUDIO_WAKE_REQUIRED=false: every authenticated audio turn is answered, no wake word needed.
    action, question, awake = gate_audio_turn(
        "what were the vitals", wake_required=False, awake_until=0.0, now=200.0)
    assert action == "answer" and question is None  # answered as-is
    assert awake == 0.0                              # window logic irrelevant when disabled
