"""LiveKit agent worker — offline coverage of the wiring + speech/handler bridges.

The worker itself needs a live LiveKit room (and is marked `# pragma: no cover`); here we verify
everything that *can* be checked without infra: config -> WorkerOptions mapping, the not-configured
guard, transcript extraction, and the STT/TTS adapter bridges driven against the real SDK types
with fake adapters/emitters (no Whisper/Piper models, no audio hardware).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

# The whole module is meaningless without the livekit extra; skip cleanly if it's absent.
pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402

from common.config import DEFAULT  # noqa: E402
from voice.adapters import STTAdapter, TTSAdapter  # noqa: E402
from voice.handlers import EchoHandler  # noqa: E402
from livekit.agents import StopResponse  # noqa: E402
from livekit.agents import llm as lkllm  # noqa: E402

from voice.livekit_agent import (  # noqa: E402
    build_handler,
    build_worker_options,
    emit_tts_audio,
    last_user_text,
    make_agent_class,
    make_speech_bridges,
    make_stub_llm,
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


def test_stub_llm_satisfies_pipeline_gate():
    # AgentSession skips reply generation when llm is None (it `return`s in on_end_of_turn), so the
    # overridden llm_node never runs. The stub flips `llm is not None` without being a RealtimeModel
    # and without ever generating: chat() must raise if anything tries to call it directly.
    from livekit.agents import llm as lkllm

    stub = make_stub_llm()
    assert isinstance(stub, lkllm.LLM)
    assert not isinstance(stub, lkllm.RealtimeModel)
    with pytest.raises(RuntimeError, match="must not be called"):
        stub.chat(chat_ctx=None)


def test_build_handler_echo():
    handler, greeting, cleanup = build_handler("echo")
    assert isinstance(handler, EchoHandler)
    assert greeting is None
    assert cleanup is None
    assert handler.respond("ping", session_id="s") == "ping"


# --- wake-word gate (on_user_turn_completed) ---


class _FakeAuthHandler:
    """Stand-in handler: toggleable auth state + an echo respond()."""

    def __init__(self, authenticated: bool) -> None:
        self._authed = authenticated

    def is_authenticated(self, session_id: str) -> bool:
        return self._authed

    def respond(self, text, *, session_id):
        return f"echo:{text}"


def _make_agent(authenticated: bool):
    HandlerAgent = make_agent_class("hey vios", awake_window_s=30.0)
    return HandlerAgent(_FakeAuthHandler(authenticated), session_id="room1", greeting=None)


def _turn(agent, text):
    """Run on_user_turn_completed for an audio turn; return (raised_stop, message)."""
    msg = lkllm.ChatMessage(role="user", content=[text])
    try:
        asyncio.run(agent.on_user_turn_completed(lkllm.ChatContext(), msg))
        return False, msg
    except StopResponse:
        return True, msg


def test_wake_gate_skipped_before_auth():
    # Pre-auth (PIN/alert/ack): every audio turn passes through, no wake word needed.
    agent = _make_agent(authenticated=False)
    stopped, msg = _turn(agent, "one two three four")
    assert stopped is False
    assert msg.text_content == "one two three four"  # unchanged


def test_wake_gate_drops_unprompted_audio_after_auth():
    # Post-auth with no wake word and not awake -> dropped (noise / hallucination).
    agent = _make_agent(authenticated=True)
    assert _turn(agent, "it's been a lot of years")[0] is True


def test_wake_word_opens_turn_and_strips_phrase():
    agent = _make_agent(authenticated=True)
    stopped, msg = _turn(agent, "hey vios what were the vitals")
    assert stopped is False
    assert msg.text_content == "what were the vitals"  # wake phrase stripped for the LLM
    assert agent._awake_until > 0.0  # now awake


def test_follow_up_within_awake_window_passes_without_wake_word():
    agent = _make_agent(authenticated=True)
    agent._awake_until = time.monotonic() + 30.0  # simulate a recent wake word
    stopped, msg = _turn(agent, "and the heart rate")
    assert stopped is False
    assert msg.text_content == "and the heart rate"


def test_bare_wake_word_arms_but_drops_empty_turn():
    agent = _make_agent(authenticated=True)
    stopped, _ = _turn(agent, "hey vios")
    assert stopped is True              # nothing to answer yet
    assert agent._awake_until > 0.0     # but now awake for the follow-up


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
