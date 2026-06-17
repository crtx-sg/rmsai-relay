"""LiveKit agent worker — offline coverage of the wiring + speech/handler bridges.

The worker itself needs a live LiveKit room (and is marked `# pragma: no cover`); here we verify
everything that *can* be checked without infra: config -> WorkerOptions mapping, the not-configured
guard, transcript extraction, and the STT/TTS adapter bridges driven against the real SDK types
with fake adapters/emitters (no Whisper/Piper models, no audio hardware).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

# The whole module is meaningless without the livekit extra; skip cleanly if it's absent.
pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402

from common.config import DEFAULT  # noqa: E402
from voice.adapters import STTAdapter, TTSAdapter  # noqa: E402
from voice.handlers import EchoHandler  # noqa: E402
from voice.livekit_agent import (  # noqa: E402
    build_handler,
    build_worker_options,
    emit_tts_audio,
    last_user_text,
    make_speech_bridges,
    run_agent,
)

_UNCONFIGURED = replace(DEFAULT, livekit_url="", livekit_api_key="", livekit_api_secret="")
_CONFIGURED = replace(DEFAULT, livekit_url="wss://demo.livekit.cloud",
                      livekit_api_key="APIabc", livekit_api_secret="s" * 40)


# --- wiring ---


def test_build_worker_options_maps_config():
    opts = build_worker_options(_CONFIGURED)
    assert opts.ws_url == "wss://demo.livekit.cloud"
    assert opts.api_key == "APIabc"
    assert opts.api_secret == "s" * 40
    assert callable(opts.entrypoint_fnc)


def test_run_agent_requires_livekit_config():
    with pytest.raises(SystemExit, match="LiveKit is not configured"):
        run_agent(_UNCONFIGURED)


def test_build_handler_echo():
    handler, greeting, cleanup = build_handler("echo")
    assert isinstance(handler, EchoHandler)
    assert greeting is None
    assert cleanup is None
    assert handler.respond("ping", session_id="s") == "ping"


# --- transcript extraction ---


class _Msg:
    def __init__(self, role, text):
        self.role = role
        self.text_content = text


class _Ctx:
    def __init__(self, items):
        self.items = items


def test_last_user_text_returns_latest_user_turn():
    ctx = _Ctx([_Msg("user", "first"), _Msg("assistant", "reply"), _Msg("user", "second")])
    assert last_user_text(ctx) == "second"


def test_last_user_text_empty_when_no_user_turn():
    assert last_user_text(_Ctx([_Msg("assistant", "hi")])) == ""
    assert last_user_text(_Ctx([])) == ""


# --- STT bridge: AudioFrame -> WAV -> adapter -> SpeechEvent ---


class _RecordingSTT(STTAdapter):
    def __init__(self, transcript):
        self.transcript = transcript
        self.received: bytes | None = None

    def transcribe(self, audio: bytes) -> str:
        self.received = audio
        return self.transcript


def test_local_stt_bridges_adapter():
    from livekit.agents import stt as lkstt

    LocalSTT, _ = make_speech_bridges()
    rec = _RecordingSTT("acknowledged copy")
    samples = 1600  # 0.1s @ 16 kHz, silence
    frame = rtc.AudioFrame(data=b"\x00\x00" * samples, sample_rate=16000,
                           num_channels=1, samples_per_channel=samples)

    event = asyncio.run(LocalSTT(rec)._recognize_impl([frame]))

    assert event.type == lkstt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "acknowledged copy"
    assert rec.received is not None and rec.received[:4] == b"RIFF"  # adapter got a WAV buffer


# --- TTS bridge: adapter -> AudioEmitter ---


class _FakeEmitter:
    def __init__(self):
        self.init: dict | None = None
        self.pushed: list[bytes] = []
        self.flushed = False

    def initialize(self, **kw):
        self.init = kw

    def push(self, data):
        self.pushed.append(data)

    def flush(self):
        self.flushed = True


class _WavTTS(TTSAdapter):
    def synthesize(self, text: str) -> bytes:
        return b"RIFF" + text.encode()


class _PcmTTS(TTSAdapter):
    def synthesize(self, text: str) -> bytes:  # not used when pcm_stream exists
        return b""

    def pcm_stream(self, text: str):
        yield b"\x01\x02"
        yield b"\x03\x04"


def test_emit_tts_audio_wav_path():
    em = _FakeEmitter()
    asyncio.run(emit_tts_audio(_WavTTS(), 22050, "hello", em))
    assert em.init["mime_type"] == "audio/wav"
    assert em.init["sample_rate"] == 22050
    assert em.pushed == [b"RIFFhello"]
    assert em.flushed


def test_emit_tts_audio_pcm_stream_path():
    em = _FakeEmitter()
    asyncio.run(emit_tts_audio(_PcmTTS(), 16000, "hello", em))
    assert em.init["mime_type"] == "audio/pcm"
    assert em.init.get("stream", False) is False  # single segment: no start_segment() framing
    assert em.pushed == [b"\x01\x02", b"\x03\x04"]  # one push per sentence chunk
    assert em.flushed


def test_local_tts_capabilities():
    _, LocalTTS = make_speech_bridges()
    lt = LocalTTS(_WavTTS(), sample_rate=22050)
    assert lt.sample_rate == 22050
    assert lt.capabilities.streaming is False
