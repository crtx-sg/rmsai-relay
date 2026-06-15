"""Phase 5 voice echo bot: round-trip, barge-in, latency (stub adapters, no audio hardware)."""

from __future__ import annotations

from voice.adapters import StubSTT, StubTTS
from voice.handlers import EchoHandler
from voice.session import TurnMetrics, VoiceSession


def _session() -> VoiceSession:
    return VoiceSession(StubSTT(), StubTTS(), EchoHandler(), "call-1")


# --- adapter codec round-trip ---


def test_tts_audio_round_trips_through_stt():
    audio = StubTTS().synthesize("status of bed three")
    assert StubSTT().transcribe(audio) == "status of bed three"


# --- echo loop: speak X -> hear X back ---


def test_echo_loop_speak_and_hear_back():
    spoken_by_caller = StubTTS().synthesize("ventricular tachycardia on bed five")
    result = _session().handle_turn(spoken_by_caller)
    assert result.heard == "ventricular tachycardia on bed five"
    assert result.spoken == "ventricular tachycardia on bed five"
    # what the caller hears back, re-transcribed, matches what they said (parrot)
    assert StubSTT().transcribe(result.audio) == "ventricular tachycardia on bed five"
    assert not result.interrupted


# --- barge-in ---


def test_barge_in_interrupts_playback():
    caller = StubTTS().synthesize("tell me everything about atrial fibrillation rate control now")
    result = _session().handle_turn(caller, barge_in_after=2)
    assert result.interrupted
    assert result.chunks_played == 2
    # playback was truncated -> not the full response
    assert StubSTT().transcribe(result.audio) != result.spoken
    assert len(StubSTT().transcribe(result.audio).split()) == 2


def test_no_barge_in_plays_full_response():
    caller = StubTTS().synthesize("one two three four")
    result = _session().handle_turn(caller)
    assert result.chunks_played == 4
    assert not result.interrupted


# --- latency ---


def test_latency_recorded_and_ordered():
    result = _session().handle_turn(StubTTS().synthesize("hello there"))
    assert result.metrics.first_audio_ms >= 0.0
    assert result.metrics.full_turn_ms >= result.metrics.first_audio_ms


def test_within_budget_helper():
    m = TurnMetrics(first_audio_ms=50.0, full_turn_ms=200.0)
    assert m.within_budget(first_audio_ms=100.0, full_turn_ms=500.0)
    assert not m.within_budget(first_audio_ms=10.0, full_turn_ms=500.0)
