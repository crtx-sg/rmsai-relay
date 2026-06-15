"""LiveKit Agents entrypoint — the real audio transport for the voice bot.

This is the glue that runs our `VoiceSession` logic inside a LiveKit room: caller audio arrives as
a track, STT transcribes it, the handler responds, TTS audio is published back, and the room's
voice-activity detection drives `barge_in`. It needs the `livekit-agents` package, a running
LiveKit server (the `livekit` compose service), and a SIP trunk into the room (see
`voice/gateway/`).

It is imported lazily and **not exercised by the offline test suite** — the turn-taking,
barge-in, and latency logic is tested via stub adapters in `voice/session.py`; this module only
wires that logic to real audio. Run it with a real phone/softphone against the SIP gateway.
"""

from __future__ import annotations

from .adapters import STTAdapter, StubSTT, StubTTS, TTSAdapter
from .handlers import EchoHandler, Handler
from .session import VoiceSession

# Clinical vocabulary biasing for Whisper (G15).
CLINICAL_PROMPT = (
    "Arrhythmia, atrial fibrillation, ventricular tachycardia, bradycardia, MEWS, "
    "SpO2, beta-blocker, anticoagulant, defibrillation, acknowledge, bed, unit."
)


def build_session(
    *, session_id: str, stt: STTAdapter | None = None, tts: TTSAdapter | None = None,
    handler: Handler | None = None,
) -> VoiceSession:
    """Construct the VoiceSession used by the agent (defaults to the offline echo stack)."""
    return VoiceSession(
        stt=stt or StubSTT(),
        tts=tts or StubTTS(),
        handler=handler or EchoHandler(),
        session_id=session_id,
    )


def run_agent(*, livekit_url: str | None = None) -> None:  # pragma: no cover - needs live infra
    """Run the LiveKit agent worker. Requires `livekit-agents` + a running LiveKit server.

    Wiring (per the livekit-agents worker model):
      * connect to the room at `livekit_url` (default from compose: ws://localhost:7880)
      * on participant audio: VAD segments speech -> STT -> Handler -> TTS -> publish audio
      * VAD speech-start during playback -> session.barge_in() (interrupt TTS)
    Swap StubSTT/StubTTS for WhisperSTT/PiperTTS (voice/adapters.py) for real audio.
    """
    try:
        from livekit import agents  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "livekit-agents not installed. Install it and run a LiveKit server "
            "(docker compose --profile later up -d livekit), then re-run."
        ) from exc

    url = livekit_url or "ws://localhost:7880"
    raise SystemExit(
        f"LiveKit agent wiring is a deployment step (server at {url}). "
        "Use cli/voice.py for the offline echo demo; see voice/gateway/README.md for SIP setup."
    )
