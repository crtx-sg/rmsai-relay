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
import logging
import os
import time
from functools import partial

from common.config import DEFAULT, Config

from .adapters import STTAdapter, TTSAdapter, build_stt, build_tts, speakable
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

# Third-party libraries that flood the worker console at DEBUG (presidio dumps every PII recognizer
# decision per turn; whisper/piper log per-segment phonemes; neo4j/redis/http log every call). We
# keep our own `[worker]`/`[voice]` prints and livekit's own logs; these get pinned to WARNING.
# `cli.voice_worker dev` sets the root logger to DEBUG, so quieting must be explicit per-logger.
_NOISY_LOGGERS = (
    "faster_whisper", "piper", "piper.voice",
    "neo4j", "neo4j.pool", "h5py", "h5py._conv", "httpx", "httpcore", "urllib3", "redis",
)
# Presidio logs a WARNING per turn for every NER label it doesn't map ("Entity CARDINAL/MONEY/PERCENT
# is not mapped…") — pure noise that buries the real log. Pin these to ERROR.
_VERY_NOISY_LOGGERS = ("presidio-analyzer", "presidio-anonymizer")


def quiet_noisy_loggers(level: int = logging.WARNING) -> None:
    """Pin chatty third-party loggers so the worker console stays readable."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for name in _VERY_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


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
            # Log both ends of the STT call: reaching here proves audio flowed all the way through
            # VAD segmentation, and the returned text separates "heard nothing" (silence/mic) from
            # "backend returned nothing" (STT). Duration is the audio the segment actually carried.
            print(f"[worker] stt: transcribing {frame.duration:.2f}s of audio "
                  f"({self._adapter.__class__.__name__})", flush=True)
            text = await loop.run_in_executor(None, self._adapter.transcribe, wav)
            print(f"[worker] stt: -> {text!r}", flush=True)
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
            # Single TTS boundary for the live call: normalize machine tokens (e.g. NORMAL_SINUS ->
            # "NORMAL SINUS") so no underscore is spoken, across greeting/select/Q&A and any backend.
            return LocalTTSStream(tts=self, input_text=speakable(text), conn_options=conn_options)

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


def make_agent_class(wake_word: str = "hey vios", awake_window_s: float = 30.0,
                     wake_required: bool = True):
    """Define and return a `HandlerAgent` whose 'LLM' node is our conversation `Handler` (lazy).

    `wake_word`/`awake_window_s` gate follow-up *audio* Q&A: after the alert (once the session is
    authenticated), an audio turn is only answered if it starts with the wake word or arrives within
    `awake_window_s` of the last wake word. Set `wake_required=False` (config `AUDIO_WAKE_REQUIRED`)
    to answer every authenticated audio turn — the escape hatch when STT mishears the brand word.
    PIN entry, the spoken alert, and the verbal ack run before auth and are never gated; text-chat
    turns bypass this hook entirely.
    """
    from livekit.agents import Agent, StopResponse  # noqa: PLC0415

    from .wake import gate_audio_turn  # noqa: PLC0415

    class HandlerAgent(Agent):
        """LiveKit `Agent` that answers via a `Handler` instead of an LLM (no LLM in the path)."""

        def __init__(
            self, handler: Handler, session_id: str, greeting: str | None,
            *, push_to_talk: bool = False,
        ) -> None:
            super().__init__(instructions="")  # unused: llm_node is overridden
            self._handler = handler
            self._session_id = session_id
            self._greeting = greeting
            self._push_to_talk = push_to_talk  # in-app chat: app controls the mic, so no wake gate
            self._awake_until = 0.0  # monotonic deadline; audio Q&A is open until then

        async def on_enter(self) -> None:
            if self._greeting:
                print(f"[worker] speaking greeting: {self._greeting!r}", flush=True)
                self.session.say(self._greeting)

        async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
            """Wake-word gate for follow-up audio (raise StopResponse to drop a turn silently)."""
            if self._push_to_talk:
                return  # push-to-talk (in-app): every captured turn is intended, answer it
            is_auth = getattr(self._handler, "is_authenticated", None)
            # Only the post-alert (authenticated) Q&A phase is gated. PIN/alert/ack pass through.
            if not (is_auth and is_auth(self._session_id)):
                return
            text = (new_message.text_content or "").strip()
            action, question, self._awake_until = gate_audio_turn(
                text, wake_word=wake_word, wake_required=wake_required,
                awake_until=self._awake_until, now=time.monotonic(), awake_window_s=awake_window_s,
            )
            if action == "drop":
                print(f"[worker] dropping audio turn (wake gate): {text!r}", flush=True)
                raise StopResponse()
            if question is not None:
                new_message.content = [question]  # strip wake phrase; LLM sees only the question
                print(f"[worker] wake word -> awake {awake_window_s:.0f}s; q={question!r}",
                      flush=True)

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


def _build_inbox_room(ctx, config, loop):  # pragma: no cover - needs a live LiveKit room
    """Wire the persistent inbox room: selection-scoped chat handler + inline-artifact push.

    Returns `(handler, greeting, cleanup)`. Registers a data-message listener so a `type:"select"`
    from the app scopes the conversation to that event, and gives the handler an `on_show` callback
    that mints a fresh scoped artifact token and pushes a `type:"show"` message so the app renders
    the artifact inline. Runs against the configured backends (Ollama LLM + the KB's embedder).
    """
    import json  # noqa: PLC0415

    from live.artifact_tokens import ArtifactTokenStore  # noqa: PLC0415

    from .handlers import build_inbox_handler  # noqa: PLC0415

    try:
        token_store = ArtifactTokenStore.from_config(config)
    except Exception:  # noqa: BLE001 - no redis -> chat still works, just no inline artifacts
        token_store = None

    def _on_show(event_id: str, kind: str) -> None:
        if token_store is None:
            return
        token, expires = token_store.mint(event_id, kind)
        payload = json.dumps({"type": "show", "event_id": event_id, "kind": kind,
                              "url": f"/artifact/{token}", "expires": expires}).encode("utf-8")
        # respond() runs in an executor thread; hop back to the loop to publish.
        asyncio.run_coroutine_threadsafe(
            ctx.room.local_participant.publish_data(payload, reliable=True), loop
        )
        print(f"[worker] inbox show -> {kind} for event {event_id}", flush=True)

    handler, greeting, cleanup = build_inbox_handler(
        embedder=DEFAULT.embedder, llm=DEFAULT.llm_provider, on_show=_on_show
    )

    # NOTE: in the agents worker only the lk.chat text path (text_input_cb) reliably delivers
    # inbound messages — raw data packets (room.on("data_received")) and custom text-stream topics
    # are swallowed by the framework. So the app scopes the conversation by sending a
    # "/select <event_id>" control message on lk.chat, handled in InboxHandler.respond.
    return handler, greeting, cleanup


#: Push-to-talk control messages the app sends on the `lk.chat` text channel (same proven transport
#: as `/select`; raw data packets don't reach the agent worker). They frame an audio turn explicitly.
_PTT_START, _PTT_END = "/ptt-start", "/ptt-end"


def ptt_command(text: str) -> str | None:
    """Classify a chat message as a push-to-talk control frame: `"start"`, `"end"`, or `None`.

    These are transport-level, not conversation: they must be intercepted *before* the text reaches
    `handler.respond`, or the orchestrator would answer "/ptt-end" as if it were a question.
    """
    stripped = (text or "").strip()
    if stripped == _PTT_START:
        return "start"
    if stripped == _PTT_END:
        return "end"
    return None


#: How long a freshly-connected inbox job waits for the room roster to settle before checking for a
#: duplicate agent. Two dispatches racing (gateway `/session` + the worker's auto-redispatch) join
#: within milliseconds of each other, so the check must not run on an empty roster.
_DUP_AGENT_SETTLE_S = 2.0


def duplicate_agent_should_yield(my_identity: str, agent_identities) -> bool:
    """True when *this* agent should leave because another agent is already serving the room.

    `create_agent_dispatch` is idempotent, but only against state it can see: two dispatches issued
    in the same instant (the gateway's `/session` and the worker's auto-redispatch) both look at a
    room with no agent and no job yet, so both fire. Two agents then answer every message — the
    clinician hears the selected event spoken twice and gets every chat reply twice.

    The tiebreak is the identity sort order, so both sides reach the same verdict independently and
    exactly one survives: an agent yields only to a *lower* identity. Symmetric "someone else is
    here, I'll go" would empty the room.
    """
    others = [i for i in agent_identities if i and i != my_identity]
    return any(i < my_identity for i in others)


#: Participant kinds RoomIO is willing to link its audio input to (the SDK's
#: `DEFAULT_PARTICIPANT_KINDS`): STANDARD (the app), SIP (a phone caller), CONNECTOR. Notably NOT
#: AGENT — linking to another agent would feed our own output back in.
_LINKABLE_KINDS = (0, 3, 5)


def chat_sender_allowed(kind: int | None) -> bool:
    """Whether a `lk.chat` message from a participant of this kind may drive a turn.

    Agents are excluded, and that exclusion is load-bearing: the worker answers on the same topic it
    listens to, so two agents sharing a room would each treat the other's reply as a fresh question
    and volley forever — the console fills with the same answer and the app shows it endlessly
    (observed 2026-08-19). RoomIO's own handler never hit this because it only accepted the linked
    participant; owning the topic means owning that filter too. Only humans and SIP callers speak.
    """
    return kind in _LINKABLE_KINDS


def relink_target(linked_identity: str | None, new_identity: str, kind: int) -> str | None:
    """Identity the room's audio input should switch to when `new_identity` joins, or `None`.

    RoomIO links its **audio** input to exactly one participant identity, and it does that once:
    `_init_task` awaits the first participant, calls `set_participant(identity)` and exits. The
    connect handler then early-returns for every later participant because an identity is already
    pinned. The app mints a fresh `clinician-<hex>` on every `/session`, so after any reload the
    audio input stays bound to the *disconnected* identity and the new browser's mic track is never
    subscribed — voice goes silently dead (no STT, no log) while text chat keeps working, because
    the text stream handler is room-scoped, not participant-scoped.

    Returns the new identity when a re-link is needed. One clinician at a time per inbox room is
    assumed: the newest join wins.
    """
    if kind not in _LINKABLE_KINDS or not new_identity:
        return None
    if linked_identity == new_identity:
        return None  # already linked (e.g. RoomIO's own first-participant handling won the race)
    return new_identity


def build_room_input_options(text_input_cb, *, is_inbox: bool):
    """`RoomInputOptions` for a job, with session teardown gated on the room *kind*.

    The inbox room is PERSISTENT and the app takes a **fresh identity** on every `/session` (page
    reload / re-login), so the previous `clinician-*` participant leaves CLIENT_INITIATED. The SDK
    default (`close_on_disconnect=True`) closes the `AgentSession` on that — which detaches
    `text_input_cb`, so every later chat message is dropped with *"ignoring text stream with topic
    'lk.chat', no callback attached"* and the app gets no reply. The job itself keeps running and
    the agent stays a room participant, so `create_agent_dispatch` (a deliberate no-op when an agent
    is present) can't re-wire it either: in-app chat is dead until the worker is restarted. Keeping
    the session open across reconnects fixes it — RoomIO re-links audio to the new participant, and
    the text path is room-scoped so it never needed re-linking.

    Per-event call rooms keep the default: there the caller hanging up *should* end the session.
    """
    from livekit.agents import RoomInputOptions  # noqa: PLC0415

    return RoomInputOptions(text_input_cb=text_input_cb, close_on_disconnect=not is_inbox)


async def _entrypoint(ctx) -> None:  # pragma: no cover - needs a live LiveKit room
    """Worker job: join the room, then run STT -> Handler -> TTS until the call ends."""
    from livekit.agents import AgentSession, TurnHandlingOptions  # noqa: PLC0415
    from livekit.agents.voice.room_io import TextInputEvent as _TextInputEvent  # noqa: PLC0415

    quiet_noisy_loggers()  # presidio/whisper/piper/etc. flood the console at DEBUG
    config = DEFAULT
    mode = os.environ.get("VOICE_MODE", "orchestrator")

    LocalSTT, LocalTTS = make_speech_bridges()
    HandlerAgent = make_agent_class(config.audio_wake_word, config.audio_wake_window_s,
                                    config.audio_wake_required)

    stt_adapter = build_stt(config)
    tts_adapter = build_tts(config)
    sample_rate = getattr(tts_adapter, "sample_rate", _FALLBACK_SAMPLE_RATE)

    loop = asyncio.get_running_loop()
    # The per-hospital inbox room is a persistent, long-lived room for in-app chat (text + voice),
    # scoped to the worklist event the clinician selects. Everything else keeps the per-event
    # behaviour: an outbound (relay-initiated) SIP call if an alert is staged, else an inbound query.
    is_inbox = ctx.room.name.startswith("rmsai-inbox-")
    if is_inbox:
        handler, greeting, cleanup = _build_inbox_room(ctx, config, loop)
        kind = "INBOX (in-app chat, push-to-talk)"
    else:
        try:
            alert_store = OutboundAlertStore.from_config(config)
        except Exception:  # noqa: BLE001 - no redis -> inbound-only worker still functions
            alert_store = None
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
    from livekit import rtc as _rtc  # noqa: PLC0415

    def _yield_if_duplicate(_participant=None) -> bool:
        """Leave if a lower-identity agent shares this room. Returns True if we yielded."""
        agents = [p.identity for p in ctx.room.remote_participants.values()
                  if p.kind == _rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]
        if not duplicate_agent_should_yield(ctx.room.local_participant.identity, agents):
            return False
        print(f"[worker] duplicate agent in {ctx.room.name!r} (also: {agents}) — yielding, "
              f"this job will exit so exactly one agent serves the room", flush=True)
        ctx.shutdown(reason="duplicate agent")
        return True

    if is_inbox:
        # Last line of defence against a dispatch race putting two agents in the persistent inbox
        # room (every reply doubled, and — until the sender filter above — an infinite volley).
        # Checking ONCE after connecting is not enough: agents cold-start ~45s apart, so the first
        # to arrive sees an empty room and the second may sort higher and keep serving too, leaving
        # both. Re-checking whenever an agent joins closes that hole — whatever the arrival order,
        # every agent above the lowest identity leaves, and exactly one survives.
        await asyncio.sleep(_DUP_AGENT_SETTLE_S)
        if _yield_if_duplicate():
            return
        ctx.room.on("participant_connected", _yield_if_duplicate)
    vad = (ctx.proc.userdata or {}).get("vad") or silero.VAD.load()
    session = AgentSession(
        stt=LocalSTT(stt_adapter),  # session + vad wrap the non-streaming STT for endpointing
        tts=LocalTTS(tts_adapter, sample_rate),
        vad=vad,
        # Gate-filler only: livekit skips reply generation when llm is None, so without this the
        # overridden llm_node (which calls our Handler -> ollama) would never run. See make_stub_llm.
        llm=make_stub_llm(),
        # Disable preemptive generation: it starts llm_node BEFORE on_user_turn_completed, so a
        # no-wake-word turn would still hit the KB/LLM (and could be spoken) before the wake gate
        # drops it. Off => the gate runs first; ignored turns do no work. (Option is a mapping, not
        # a bool — the SDK resolves it via `{**defaults, **config}`.)
        #
        # In the inbox the *button* delimits the turn, so end-of-turn is "manual": VAD endpointing
        # would need ~0.5s of trailing silence to fire, but the app mutes the mic the instant the
        # clinician releases the button — the SFU then stops forwarding audio, no silence ever
        # reaches the VAD, the turn never closes and STT is never invoked (symptom: a held-and-
        # spoken question produces NOTHING in the worker log). `/ptt-end` commits the turn instead.
        turn_handling=TurnHandlingOptions(
            preemptive_generation={"enabled": False},
            **({"turn_detection": "manual"} if is_inbox else {}),
        ),
    )
    # Log every finalized transcript: the one place that proves STT actually ran (and what it heard)
    # when a voice turn produces no answer. Text-chat turns don't pass through STT, so this is
    # audio-only. Pseudonymous by construction; synthetic speech only (hard rules #5/#6).
    session.on("user_input_transcribed",
               lambda ev: print(f"[worker] stt transcript: {getattr(ev, 'transcript', '')!r}"
                                f" (final={getattr(ev, 'is_final', None)})", flush=True))

    def _link_audio(participant) -> None:
        """Point RoomIO's audio input at `participant` if it isn't already (see `relink_target`)."""
        if participant is None:
            return
        try:
            room_io = session.room_io
            linked = room_io.linked_participant
            target = relink_target(linked.identity if linked else None,
                                   participant.identity, participant.kind)
            if target is None:
                return
            room_io.set_participant(target)
            print(f"[worker] audio input re-linked to {target}", flush=True)
        except Exception as exc:  # noqa: BLE001 - never let a room/text event kill the session
            print(f"[worker] audio re-link failed: {exc}", flush=True)

    async def _handle_ptt(action: str, participant=None) -> None:
        """Open/close an inbox audio turn on the app's button press/release.

        Press: link audio to whoever pressed the button, attach the audio input and drop whatever
        was buffered (a half-turn from a previous press, or the tail of our own TTS). Release:
        detach the input and `commit_user_turn()`, which pushes the silence that flushes our
        non-streaming STT (the SDK only does that when the input is detached — see
        `audio_recognition.commit_user_turn`) and then generates the reply.

        The press-time re-link is the belt to the `participant_connected` braces: it keys off the
        participant that is about to publish the mic track, so it holds even if the join event was
        missed (agent dispatched into a room the clinician was already in).

        Best-effort: a failed commit must not kill the session, it just costs one turn.
        """
        try:
            if action == "start":
                _link_audio(participant)
                session.input.set_audio_enabled(True)
                session.clear_user_turn()
                print(f"[worker] ptt: mic open (audio_enabled="
                      f"{session.input.audio_enabled}, linked="
                      f"{getattr(session.room_io.linked_participant, 'identity', None)})", flush=True)
                return
            session.input.set_audio_enabled(False)
            # The SDK's 2s default assumes a streaming cloud STT; ours is non-streaming and only
            # starts once the flush silence lands (Whisper on CPU / an ElevenLabs round trip), so a
            # short timeout would report "nothing heard" for a turn that was about to transcribe.
            transcript = await session.commit_user_turn(transcript_timeout=15.0)
            print(f"[worker] ptt: turn committed -> {transcript!r}", flush=True)
            if not transcript:
                # Nothing transcribed: the clinician gets silence otherwise, which is
                # indistinguishable from a broken worker.
                await ctx.room.local_participant.send_text(
                    "I didn't catch that — hold the button, speak, then release.", topic="lk.chat"
                )
        except Exception as exc:  # noqa: BLE001 - one dropped turn beats a dead session
            print(f"[worker] ptt: {action} failed: {exc}", flush=True)

    # Text chat (meet.livekit.io chat box) -> text-only reply. Bypasses generate_reply (and thus
    # TTS), so a typed question gets a typed answer on the chat topic, never spoken audio. This also
    # bypasses the wake-word gate (that lives in on_user_turn_completed, audio-only).
    async def _text_only_reply(sess, ev) -> None:
        text = (getattr(ev, "text", "") or "").strip()
        if not text:
            return
        # Push-to-talk framing (transport, not conversation): the button open/closes the audio turn.
        # Must be intercepted before handler.respond, which would otherwise answer the control word.
        if is_inbox and (ptt := ptt_command(text)) is not None:
            await _handle_ptt(ptt, getattr(ev, "participant", None))
            return
        print(f"[worker] text chat heard: {text!r}", flush=True)
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None, partial(handler.respond, text, session_id=ctx.room.name)
        )
        # Companion-app worklist: selecting a row voices that event's stored report aloud (in
        # addition to scoping chat). Inbox room only, gated by config; spoken via TTS (session.say),
        # which is independent of the text reply below. Fails silent (select_spoken_line -> None).
        if (is_inbox and config.inbox_speak_on_select
                and text.startswith("/select ") and hasattr(handler, "select_spoken_line")):
            event_id = text[len("/select "):].strip()
            line = await loop.run_in_executor(None, partial(handler.select_spoken_line, event_id))
            if line:
                print(f"[worker] speaking selected event {event_id}: {line!r}", flush=True)
                session.say(line)
        if not reply:  # control messages (e.g. /select) produce no chat bubble
            return
        print(f"[worker] text chat reply: {reply!r}", flush=True)
        await ctx.room.local_participant.send_text(reply, topic="lk.chat")

    # Own the `lk.chat` topic ourselves, *before* the session starts (only one handler per topic is
    # allowed, and first registration wins — RoomIO's own attempt then logs "already set, ignoring").
    # Two reasons this beats letting RoomIO deliver the text:
    #   1. Visibility. Every inbound message is logged with its sender the moment it lands, so
    #      "did the app's control frame reach the worker?" is answerable from the console instead of
    #      inferred from a missing downstream effect.
    #   2. RoomIO drops text from any participant that isn't the linked one (room_io.py:439-442).
    #      The app reconnects under a new identity on every /session, so that filter can silently
    #      swallow chat from a clinician the audio input hasn't re-linked to yet.
    _chat_tasks: set = set()

    def _on_chat_text(reader, participant_identity: str) -> None:
        participant = ctx.room.remote_participants.get(participant_identity)
        if not chat_sender_allowed(getattr(participant, "kind", None)):
            # Silently dropping this is the point: our own replies go out on this topic, so a second
            # agent's reply must never be read back as a question. Logged, not answered.
            print(f"[worker] lk.chat ignoring {participant_identity} "
                  f"(kind={getattr(participant, 'kind', None)}, not a caller)", flush=True)
            return

        async def _read() -> None:
            try:
                text = await reader.read_all()
                print(f"[worker] lk.chat rx from {participant_identity}: {text!r}", flush=True)
                await _text_only_reply(session, _TextInputEvent(
                    text=text, info=reader.info, participant=participant))
            except Exception:  # noqa: BLE001 - one bad message must not kill the chat path
                import traceback  # noqa: PLC0415

                print(f"[worker] lk.chat handler failed:\n{traceback.format_exc()}", flush=True)

        task = asyncio.create_task(_read())
        _chat_tasks.add(task)
        task.add_done_callback(_chat_tasks.discard)

    try:
        ctx.room.register_text_stream_handler("lk.chat", _on_chat_text)
        print("[worker] lk.chat handler registered (worker-owned)", flush=True)
    except ValueError as exc:  # pragma: no cover - someone else already owns the topic
        print(f"[worker] lk.chat handler NOT registered ({exc}); falling back to RoomIO's", flush=True)

    await session.start(
        agent=HandlerAgent(handler, session_id=ctx.room.name, greeting=greeting,
                           push_to_talk=is_inbox),
        room=ctx.room,
        room_input_options=build_room_input_options(_text_only_reply, is_inbox=is_inbox),
    )
    if is_inbox:
        # Start with the audio input detached: the inbox mic is push-to-talk, so nothing should be
        # recognized until the button is held. This also guarantees the input is *detached* at
        # `/ptt-end`, which is the condition the SDK requires before flushing STT on commit.
        session.input.set_audio_enabled(False)

        # Re-link the audio input when the clinician reconnects under a new identity (see
        # relink_target): without this the persistent inbox room keeps text chat alive but loses
        # voice for the rest of the session.
        ctx.room.on("participant_connected", _link_audio)


def _prewarm(proc) -> None:  # pragma: no cover - needs the silero plugin + a worker process
    quiet_noisy_loggers()  # quiet the noisy libs in each warmed job process too
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
        # Register under a name => EXPLICIT dispatch only (no auto-join of new rooms). Every room that
        # needs the agent requests it via voice.livekit_cloud.create_agent_dispatch: the gateway on
        # /session (inbox), the consumer per outbound event, and cli.livekit_token (inbound demo). An
        # empty name here would revert to automatic dispatch and its restart-ordering fragility.
        agent_name=config.livekit_agent_name,
        ws_url=config.livekit_url,
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
        # livekit-agents' health-check HTTP server. Configurable so a containerized worker on host
        # networking can coexist with one running on the host (both would bind 8081 otherwise).
        port=config.livekit_worker_http_port,
    )


def _start_auto_redispatch(config: Config) -> None:  # pragma: no cover - needs a live LiveKit server
    """Re-wire live inbox rooms to this worker after it (re)starts, without an app re-login.

    A worker registered under a name does NOT auto-join rooms; the gateway only dispatches at
    `/session`. So a restarted worker leaves already-connected inbox rooms agent-less. This runs in a
    daemon thread and, once the worker has registered, dispatches the agent into every live
    `rmsai-inbox-*` room that lacks one. Retries a few times to ride out the registration race, with a
    per-room cooldown longer than the agent's cold-start so a still-joining room isn't dispatched
    twice (which would put two agents in the room). Idempotent and best-effort.
    """
    import threading  # noqa: PLC0415

    from voice.livekit_cloud import redispatch_existing_rooms  # noqa: PLC0415

    first_delay_s, interval_s, attempts, cooldown_s = 15.0, 15.0, 8, 75.0

    def _work() -> None:
        recent: dict[str, float] = {}  # room -> monotonic time we last dispatched it
        time.sleep(first_delay_s)  # let the worker register before the first dispatch
        for _ in range(attempts):
            now = time.monotonic()
            skip = {r for r, t in recent.items() if now - t < cooldown_s}  # still cold-starting
            try:
                for room in redispatch_existing_rooms(config, skip=skip):
                    recent[room] = now
                    print(f"[worker] auto-redispatch -> room {room}", flush=True)
            except Exception as exc:  # noqa: BLE001 - a background retry must never crash the worker
                print(f"[worker] auto-redispatch error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(interval_s)

    threading.Thread(target=_work, daemon=True, name="rmsai-redispatch").start()


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
    if config.tts_backend == "elevenlabs":
        print(
            "NOTE: TTS_BACKEND=elevenlabs (cloud). Spoken text is de-identified (Presidio/regex) "
            "before being sent, on top of pseudonym-by-construction — no PHI in the outbound text."
        )
    if config.stt_backend == "elevenlabs":
        print(
            "WARNING: STT_BACKEND=elevenlabs (cloud). RAW caller AUDIO is sent to a third party and "
            "CANNOT be de-identified first — use SYNTHETIC speech ONLY, never real PHI (rules #4/#5)."
        )

    if config.livekit_redispatch_on_start:
        _start_auto_redispatch(config)

    cli.run_app(build_worker_options(config))
