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
    build_room_input_options,
    build_worker_options,
    duplicate_agent_should_yield,
    ptt_command,
    relink_target,
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


def test_worker_uses_named_explicit_dispatch():
    # A named agent => EXPLICIT dispatch only (no auto-join). Rooms must request it via
    # create_agent_dispatch (gateway/consumer/token CLI). An empty name would revert to fragile
    # automatic dispatch, so guard the name is actually set from config.
    assert build_worker_options(_CONFIGURED).agent_name == _CONFIGURED.livekit_agent_name
    assert build_worker_options(_CONFIGURED).agent_name  # non-empty


def test_inbox_session_survives_a_clinician_reconnect():
    # The app mints a NEW identity per /session, so a reload disconnects the old participant
    # (CLIENT_INITIATED). With the SDK default the AgentSession would close, detaching text_input_cb
    # -> every later chat message is dropped ("no callback attached") and, because the agent is
    # still a room participant, create_agent_dispatch won't re-wire it. The persistent inbox room
    # must therefore opt out; per-event call rooms keep the default (hang up == end of call).
    cb = object()
    inbox = build_room_input_options(cb, is_inbox=True)
    assert inbox.close_on_disconnect is False
    assert inbox.text_input_cb is cb
    assert build_room_input_options(cb, is_inbox=False).close_on_disconnect is True


def test_ptt_control_frames_are_recognised():
    # The button delimits the audio turn (the inbox runs manual end-of-turn detection), so these
    # two messages must be routed to the transport, never to handler.respond — which would answer
    # "/ptt-end" as if the clinician had asked a question.
    assert ptt_command("/ptt-start") == "start"
    assert ptt_command("  /ptt-end  ") == "end"
    assert ptt_command("/select abc") is None
    assert ptt_command("What were the vitals at the time of the event?") is None
    assert ptt_command("") is None


def test_audio_relinks_to_a_reconnected_clinician():
    # RoomIO pins its audio input to ONE identity, once. The app mints a new clinician-<hex> per
    # /session, so after a reload the mic track belongs to an identity the input isn't listening to
    # -> voice dies silently while text chat (room-scoped) keeps working.
    standard = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    assert relink_target("clinician-old", "clinician-new", standard) == "clinician-new"
    assert relink_target(None, "clinician-new", standard) == "clinician-new"
    assert relink_target("clinician-new", "clinician-new", standard) is None  # already linked
    # Never follow another agent into the room — that would feed our own TTS back into STT.
    assert relink_target("clinician-old", "rmsai-agent",
                         rtc.ParticipantKind.PARTICIPANT_KIND_AGENT) is None
    # A phone caller (SIP) is a legitimate audio source for the per-event call rooms.
    assert relink_target(None, "sip-caller", rtc.ParticipantKind.PARTICIPANT_KIND_SIP) == "sip-caller"


def test_duplicate_agent_tiebreak_keeps_exactly_one():
    # Two dispatches racing put two agents in the inbox room and every reply is doubled. Both jobs
    # run this check independently, so the verdict must be asymmetric: the higher identity yields,
    # the lower one stays. A symmetric "someone else is here, I'll leave" would empty the room.
    assert duplicate_agent_should_yield("agent-B", ["agent-A"])
    assert not duplicate_agent_should_yield("agent-A", ["agent-B"])
    assert not duplicate_agent_should_yield("agent-A", [])
    assert not duplicate_agent_should_yield("agent-A", ["agent-A"])  # itself, if ever listed
    # Three-way race still converges on the single lowest identity.
    assert not duplicate_agent_should_yield("agent-A", ["agent-B", "agent-C"])
    assert duplicate_agent_should_yield("agent-C", ["agent-A", "agent-B"])


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
