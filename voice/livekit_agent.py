"""LiveKit Agents worker — the real audio transport for the voice bot.

Wires our self-hosted speech stack and conversation logic into a LiveKit room:

    caller audio --(LiveKit)--> Whisper STT --> Handler --> Piper TTS --(LiveKit)--> caller

rmsai-relay *places* the call (`voice/outbound.py` dials the clinician's phone into a room via a
SIP trunk); this worker joins that room and drives the conversation. STT/TTS are the self-hosted
local backends (`build_stt`/`build_tts`; config `STT_BACKEND=whisper` / `TTS_BACKEND=piper`), and
silero provides the voice-activity detection that powers endpointing + barge-in. The conversation
`Handler` (Echo/Orchestrator) stands in for the "LLM" node, so no third-party LLM is in the audio
path and the PIN gate / de-id layers are unchanged from the text/CLI flow.

Needs the `livekit` extra (`uv sync --extra livekit`) plus the `voice` extra for the real models,
a running LiveKit endpoint (Cloud `wss://…` or self-hosted), and — for phone calls — a SIP trunk.

All `livekit`/`silero` imports are lazy so the base install (and the offline test suite) can import
this module without the extra. The worker itself needs live infra and is not exercised offline; the
turn-taking logic is covered via stub adapters in `voice/session.py`, and the speech/handler
*bridges* are unit-tested against the real SDK types in `tests/test_livekit_worker.py`.
"""

from __future__ import annotations

import asyncio
import os
from functools import partial

from common.config import DEFAULT, Config

from .adapters import STTAdapter, TTSAdapter, build_stt, build_tts
from .handlers import EchoHandler, Handler, build_handler
from .livekit_cloud import is_configured
from .session import VoiceSession

# LiveKit plugins register themselves at import time, and that registration MUST run on the main
# thread (prewarm/job init runs in worker threads). So import silero at module top, not lazily.
# Guarded so the base install (no `livekit` extra) can still import this module.
try:
    from livekit.plugins import silero
except ImportError:  # pragma: no cover - exercised only without the livekit extra
    silero = None

# TTS sample rate to assume when an adapter doesn't advertise one (e.g. the stub). Real Piper
# voices expose `.sample_rate`; lessac-medium is 22.05 kHz.
_FALLBACK_SAMPLE_RATE = 22050


def build_session(
    *, session_id: str, stt: STTAdapter | None = None, tts: TTSAdapter | None = None,
    handler: Handler | None = None,
) -> VoiceSession:
    """Construct the offline `VoiceSession` (STT/TTS from config) used by the echo/CLI demo."""
    return VoiceSession(
        stt=stt or build_stt(),
        tts=tts or build_tts(),
        handler=handler or EchoHandler(),
        session_id=session_id,
    )


def last_user_text(chat_ctx) -> str:
    """Return the most recent user utterance from a LiveKit `ChatContext` ('' if none)."""
    for item in reversed(getattr(chat_ctx, "items", [])):
        if getattr(item, "role", None) == "user":
            return (item.text_content or "").strip()
    return ""


async def emit_tts_audio(adapter: TTSAdapter, sample_rate: int, text: str, output_emitter) -> None:
    """Synthesize `text` and publish it to a LiveKit `AudioEmitter`.

    Prefers an adapter `pcm_stream` (raw int16 PCM per sentence, low latency — Piper); otherwise
    falls back to a single WAV blob. Synthesis runs off the event loop (the local models block).
    Factored out of the `ChunkedStream` so it can be unit-tested with a fake emitter.
    """
    from livekit.agents import utils  # noqa: PLC0415

    request_id = utils.shortuuid()
    loop = asyncio.get_running_loop()
    pcm_stream = getattr(adapter, "pcm_stream", None)
    if pcm_stream is not None:
        # stream=False: one ChunkedStream call == one segment; push all chunks, then flush.
        # (stream=True would require start_segment()/end_segment() framing per segment.)
        output_emitter.initialize(
            request_id=request_id, sample_rate=sample_rate, num_channels=1,
            mime_type="audio/pcm",
        )
        chunks = await loop.run_in_executor(None, lambda: list(pcm_stream(text)))
        for pcm in chunks:
            output_emitter.push(pcm)
    else:
        wav = await loop.run_in_executor(None, adapter.synthesize, text)
        output_emitter.initialize(
            request_id=request_id, sample_rate=sample_rate, num_channels=1, mime_type="audio/wav",
        )
        output_emitter.push(wav)
    output_emitter.flush()


def make_speech_bridges():
    """Define and return `(LocalSTT, LocalTTS)` bridging our adapters to LiveKit plugins.

    Lazily defined (subclassing the SDK base classes requires the import) so importing this module
    never depends on the `livekit` extra. `LocalSTT` is non-streaming — the agent session wraps it
    with the VAD for endpointing; `LocalTTS` streams raw PCM per sentence when the adapter offers a
    `pcm_stream` (Piper), else falls back to a single WAV blob.
    """
    from livekit import rtc  # noqa: PLC0415
    from livekit.agents import stt as lkstt  # noqa: PLC0415
    from livekit.agents import tts as lktts  # noqa: PLC0415
    from livekit.agents.types import (  # noqa: PLC0415
        DEFAULT_API_CONNECT_OPTIONS,
        NOT_GIVEN,
    )

    class LocalSTT(lkstt.STT):
        """Bridge a self-hosted `STTAdapter` (e.g. Whisper) into a LiveKit non-streaming STT."""

        def __init__(self, adapter: STTAdapter) -> None:
            super().__init__(
                capabilities=lkstt.STTCapabilities(streaming=False, interim_results=False)
            )
            self._adapter = adapter

        async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
            frame = rtc.combine_audio_frames(buffer)
            wav = frame.to_wav_bytes()  # WAV so faster-whisper can decode the buffer
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._adapter.transcribe, wav)
            lang = language if isinstance(language, str) else "en"
            return lkstt.SpeechEvent(
                type=lkstt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lkstt.SpeechData(language=lang, text=text)],
            )

    class LocalTTSStream(lktts.ChunkedStream):
        async def _run(self, output_emitter) -> None:
            await emit_tts_audio(
                self._tts._adapter, self._tts.sample_rate, self._input_text, output_emitter,
            )

    class LocalTTS(lktts.TTS):
        """Bridge a self-hosted `TTSAdapter` (e.g. Piper) into a LiveKit TTS."""

        def __init__(self, adapter: TTSAdapter, sample_rate: int) -> None:
            super().__init__(
                capabilities=lktts.TTSCapabilities(streaming=False),
                sample_rate=sample_rate, num_channels=1,
            )
            self._adapter = adapter

        def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
            return LocalTTSStream(tts=self, input_text=text, conn_options=conn_options)

    return LocalSTT, LocalTTS


def make_agent_class():
    """Define and return a `HandlerAgent` whose 'LLM' node is our conversation `Handler` (lazy)."""
    from livekit.agents import Agent  # noqa: PLC0415

    class HandlerAgent(Agent):
        """LiveKit `Agent` that answers via a `Handler` instead of an LLM (no LLM in the path)."""

        def __init__(self, handler: Handler, session_id: str, greeting: str | None) -> None:
            super().__init__(instructions="")  # unused: llm_node is overridden
            self._handler = handler
            self._session_id = session_id
            self._greeting = greeting

        async def on_enter(self) -> None:
            if self._greeting:
                self.session.say(self._greeting)

        async def llm_node(self, chat_ctx, tools, model_settings):
            text = last_user_text(chat_ctx)
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(
                None, partial(self._handler.respond, text, session_id=self._session_id)
            )
            yield reply

    return HandlerAgent


async def _entrypoint(ctx) -> None:  # pragma: no cover - needs a live LiveKit room
    """Worker job: join the room, then run STT -> Handler -> TTS until the call ends."""
    from livekit.agents import AgentSession  # noqa: PLC0415

    config = DEFAULT
    mode = os.environ.get("VOICE_MODE", "orchestrator")

    LocalSTT, LocalTTS = make_speech_bridges()
    HandlerAgent = make_agent_class()

    stt_adapter = build_stt(config)
    tts_adapter = build_tts(config)
    sample_rate = getattr(tts_adapter, "sample_rate", _FALLBACK_SAMPLE_RATE)

    handler, greeting, cleanup = build_handler(mode)
    if cleanup is not None:
        async def _shutdown() -> None:
            cleanup()
        ctx.add_shutdown_callback(_shutdown)

    await ctx.connect()
    vad = (ctx.proc.userdata or {}).get("vad") or silero.VAD.load()
    session = AgentSession(
        stt=LocalSTT(stt_adapter),  # session + vad wrap the non-streaming STT for endpointing
        tts=LocalTTS(tts_adapter, sample_rate),
        vad=vad,
    )
    await session.start(
        agent=HandlerAgent(handler, session_id=ctx.room.name, greeting=greeting),
        room=ctx.room,
    )


def _prewarm(proc) -> None:  # pragma: no cover - needs the silero plugin + a worker process
    proc.userdata["vad"] = silero.VAD.load()


def build_worker_options(config: Config | None = None):
    """Build `WorkerOptions` for the agent worker (LiveKit connection comes from config)."""
    from livekit.agents import WorkerOptions  # noqa: PLC0415

    config = config or DEFAULT
    return WorkerOptions(
        entrypoint_fnc=_entrypoint,
        prewarm_fnc=_prewarm,
        ws_url=config.livekit_url,
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
    )


def run_agent(config: Config | None = None) -> None:  # pragma: no cover - needs live infra
    """Run the LiveKit agent worker (delegates to `livekit.agents.cli`; pass a subcommand).

    Connection comes from config: `LIVEKIT_URL` (Cloud `wss://<project>.livekit.cloud` or
    self-hosted), `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`. Run via the CLI harness, e.g.::

        uv run python -m cli.voice_worker dev      # autoreload, connect to LiveKit
        uv run python -m cli.voice_worker start     # production worker

    Set `VOICE_MODE=echo` to use the parrot handler (loopback test); default is `orchestrator`.
    """
    config = config or DEFAULT
    if not is_configured(config):
        raise SystemExit(
            "LiveKit is not configured. Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET "
            "(LiveKit Cloud: wss://<project>.livekit.cloud) in your .env."
        )
    try:
        from livekit.agents import cli  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "livekit-agents not installed: `uv sync --extra livekit`, then re-run."
        ) from exc

    if config.stt_backend == "stub" or config.tts_backend == "stub":
        print(
            "WARNING: STT_BACKEND/TTS_BACKEND is 'stub' — the worker will not produce real audio. "
            "Set STT_BACKEND=whisper and TTS_BACKEND=piper (uv sync --extra voice) for real calls."
        )

    cli.run_app(build_worker_options(config))
