"""Offline speech round-trip CLI — plumbing tested deterministically via the stub adapters.

The real Piper->Whisper path needs models + the voice extra (run `cli.speech_check` for that);
here we only check that `round_trip` chains TTS into STT correctly, using the stub codec which is
designed to round-trip exactly.
"""

from __future__ import annotations

from cli.speech_check import round_trip
from voice.adapters import StubSTT, StubTTS


def test_round_trip_chains_tts_into_stt():
    heard, audio = round_trip("acknowledged copy", stt=StubSTT(), tts=StubTTS())
    assert heard == "acknowledged copy"  # stub codec round-trips exactly
    assert isinstance(audio, bytes) and audio  # non-empty synthesized audio
