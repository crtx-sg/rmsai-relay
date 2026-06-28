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
import time
from functools import partial

from common.config import DEFAULT, Config

from .adapters import STTAdapter, TTSAdapter, build_stt, build_tts
from .handlers import EchoHandler, Handler, build_handler, build_outbound_handler
from .livekit_cloud import is_configured
from .outbound_alert import OutboundAlertStore
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


def _has_alert(alert_store, room_name: str) -> bool:
    """Peek whether an outbound alert is staged for a room (for logging; does not consume it)."""
    try:
        return alert_store.get(room_name) is not None
    except Exception:  # noqa: BLE001
        return False


def resolve_handler(room_name: str, alert_store, *, mode: str = "orchestrator"):
    """Pick the handler for a room: outbound (event-seeded) if an alert is waiting, else generic.

    The relay writes an `OutboundAlert` keyed by room name before placing an outbound call, so a
    room with a pending alert is a relay-initiated call → speak that event after the PIN gate. A
    room with no alert is an inbound call → the standard PIN-gated orchestrator handler. Returns
    `(handler, greeting, cleanup)`; the alert is consumed (deleted) so a retry/redial is explicit.
    """
    alert = None
    if alert_store is not None:
        try:
            alert = alert_store.get(room_name)
        except Exception:  # noqa: BLE001 - a store hiccup must not block answering the call
            alert = None
    # Live voice runs against the configured backends (Ollama LLM + the embedder the KB was
    # indexed with), not the terminal demo's offline echo/hashing defaults.
    llm = DEFAULT.llm_provider
    embedder = DEFAULT.embedder
    if alert is not None:
        handler, greeting, cleanup = build_outbound_handler(alert, embedder=embedder, llm=llm)
        try:
            alert_store.delete(room_name)
        except Exception:  # noqa: BLE001
            pass
        return handler, greeting, cleanup
    return build_handler(mode, embedder=embedder, llm=llm)


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


def make_stub_llm():
    """Return a no-op `LLM` that satisfies the AgentSession pipeline gate (lazy).

    livekit-agents skips reply generation entirely when `session.llm is None` — `AgentActivity`'s
    end-of-turn handler hits `elif self.llm is None: return` and never calls the agent's `llm_node`.
    Our conversation `Handler` *is* the 'LLM' (see `HandlerAgent.llm_node`, which routes to
    `handler.respond()` -> orchestrator -> configured provider), so this stub exists only to flip
    `llm is not None`. Its `chat()` is never invoked because `llm_node` is overridden, and no
    `llm.capabilities` access is on the non-realtime/no-tools path we drive.
    """
    from livekit.agents import llm as lkllm  # noqa: PLC0415

    class _StubLLM(lkllm.LLM):
        def chat(self, *args, **kwargs):  # pragma: no cover - never called (llm_node is overridden)
            raise RuntimeError(
                "stub LLM.chat() must not be called; HandlerAgent.llm_node handles replies"
            )

    return _StubLLM()


def make_agent_class(wake_word: str = "hey vios", awake_window_s: float = 30.0):
    """Define and return a `HandlerAgent` whose 'LLM' node is our conversation `Handler` (lazy).

    `wake_word`/`awake_window_s` gate follow-up *audio* Q&A: after the alert (once the session is
    authenticated), an audio turn is only answered if it starts with the wake word or arrives within
    `awake_window_s` of the last wake word. PIN entry, the spoken alert, and the verbal ack run
    before auth and are never gated; text-chat turns bypass this hook entirely.
    """
    from livekit.agents import Agent, StopResponse  # noqa: PLC0415

    from .wake import detect_wake_word  # noqa: PLC0415

    class HandlerAgent(Agent):
        """LiveKit `Agent` that answers via a `Handler` instead of an LLM (no LLM in the path)."""

        def __init__(self, handler: Handler, session_id: str, greeting: str | None) -> None:
            super().__init__(instructions="")  # unused: llm_node is overridden
            self._handler = handler
            self._session_id = session_id
            self._greeting = greeting
            self._awake_until = 0.0  # monotonic deadline; audio Q&A is open until then

        async def on_enter(self) -> None:
            if self._greeting:
                print(f"[worker] speaking greeting: {self._greeting!r}", flush=True)
                self.session.say(self._greeting)

        async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
            """Wake-word gate for follow-up audio (raise StopResponse to drop a turn silently)."""
            is_auth = getattr(self._handler, "is_authenticated", None)
            # Only the post-alert (authenticated) Q&A phase is gated. PIN/alert/ack pass through.
            if not (is_auth and is_auth(self._session_id)):
                return
            text = (new_message.text_content or "").strip()
            now = time.monotonic()
            matched, remainder = detect_wake_word(text, wake_word)
            if matched:
                self._awake_until = now + awake_window_s
                if remainder:
                    new_message.content = [remainder]  # strip wake phrase; LLM sees the question
                    print(f"[worker] wake word -> awake {awake_window_s:.0f}s; q={remainder!r}",
                          flush=True)
                    return
                print("[worker] wake word (no question) -> listening for follow-up", flush=True)
                raise StopResponse()  # nothing to answer yet, but now awake
            if now < self._awake_until:
                self._awake_until = now + awake_window_s  # refresh window on each follow-up
                return
            print(f"[worker] ignoring audio (no wake word): {text!r}", flush=True)
            raise StopResponse()

        async def llm_node(self, chat_ctx, tools, model_settings):
            text = last_user_text(chat_ctx)
            print(f"[worker] heard: {text!r}", flush=True)
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(
                None, partial(self._handler.respond, text, session_id=self._session_id)
            )
            print(f"[worker] reply: {reply!r}", flush=True)
            yield reply

    return HandlerAgent


async def _entrypoint(ctx) -> None:  # pragma: no cover - needs a live LiveKit room
    """Worker job: join the room, then run STT -> Handler -> TTS until the call ends."""
    from livekit.agents import AgentSession, RoomInputOptions  # noqa: PLC0415

    config = DEFAULT
    mode = os.environ.get("VOICE_MODE", "orchestrator")

    LocalSTT, LocalTTS = make_speech_bridges()
    HandlerAgent = make_agent_class(config.audio_wake_word, config.audio_wake_window_s)

    stt_adapter = build_stt(config)
    tts_adapter = build_tts(config)
    sample_rate = getattr(tts_adapter, "sample_rate", _FALLBACK_SAMPLE_RATE)

    try:
        alert_store = OutboundAlertStore.from_config(config)
    except Exception:  # noqa: BLE001 - no redis -> inbound-only worker still functions
        alert_store = None
    # Outbound (relay-initiated) call if an alert is waiting for this room; else inbound.
    kind = "OUTBOUND (event alert staged)" if (
        alert_store is not None and _has_alert(alert_store, ctx.room.name)
    ) else "INBOUND (KB query)"
    handler, greeting, cleanup = resolve_handler(ctx.room.name, alert_store, mode=mode)
    if cleanup is not None:
        async def _shutdown() -> None:
            cleanup()
        ctx.add_shutdown_callback(_shutdown)

    await ctx.connect()
    print(f"[worker] joined room {ctx.room.name!r} over WebRTC — {kind}", flush=True)
    vad = (ctx.proc.userdata or {}).get("vad") or silero.VAD.load()
    session = AgentSession(
        stt=LocalSTT(stt_adapter),  # session + vad wrap the non-streaming STT for endpointing
        tts=LocalTTS(tts_adapter, sample_rate),
        vad=vad,
        # Gate-filler only: livekit skips reply generation when llm is None, so without this the
        # overridden llm_node (which calls our Handler -> ollama) would never run. See make_stub_llm.
        llm=make_stub_llm(),
    )
    # Text chat (meet.livekit.io chat box) -> text-only reply. Bypasses generate_reply (and thus
    # TTS), so a typed question gets a typed answer on the chat topic, never spoken audio. This also
    # bypasses the wake-word gate (that lives in on_user_turn_completed, audio-only).
    async def _text_only_reply(sess, ev) -> None:
        text = (getattr(ev, "text", "") or "").strip()
        if not text:
            return
        print(f"[worker] text chat heard: {text!r}", flush=True)
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None, partial(handler.respond, text, session_id=ctx.room.name)
        )
        print(f"[worker] text chat reply: {reply!r}", flush=True)
        await ctx.room.local_participant.send_text(reply, topic="lk.chat")

    await session.start(
        agent=HandlerAgent(handler, session_id=ctx.room.name, greeting=greeting),
        room=ctx.room,
        room_input_options=RoomInputOptions(text_input_cb=_text_only_reply),
    )


def _prewarm(proc) -> None:  # pragma: no cover - needs the silero plugin + a worker process
    proc.userdata["vad"] = silero.VAD.load()
    # Warm the Ollama model into memory so the first clinician turn doesn't pay a cold load.
    # MUST be non-blocking: prewarm runs inside LiveKit's ~10s process-init budget, and a cold
    # model load (~11s) would blow it and get the job killed (SIGUSR1). Warm in a daemon thread so
    # prewarm returns immediately and the model loads alongside the call.
    if DEFAULT.llm_provider == "ollama":
        import threading  # noqa: PLC0415

        def _warm() -> None:
            try:
                from common.providers import get_llm_provider  # noqa: PLC0415

                get_llm_provider("ollama", DEFAULT).generate("ready?")
            except Exception:  # noqa: BLE001 - warmup is best-effort; the call still works cold
                pass

        threading.Thread(target=_warm, daemon=True).start()


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
