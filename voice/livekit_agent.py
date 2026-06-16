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

from .adapters import STTAdapter, TTSAdapter, build_stt, build_tts
from .handlers import EchoHandler, Handler
from .session import VoiceSession

# STT/TTS (incl. the Whisper clinical-vocab prompt, G15) are configured via common.config and
# applied by build_stt()/build_tts().


def build_session(
    *, session_id: str, stt: STTAdapter | None = None, tts: TTSAdapter | None = None,
    handler: Handler | None = None,
) -> VoiceSession:
    """Construct the VoiceSession used by the agent (STT/TTS from config: STT_BACKEND/TTS_BACKEND)."""
    return VoiceSession(
        stt=stt or build_stt(),
        tts=tts or build_tts(),
        handler=handler or EchoHandler(),
        session_id=session_id,
    )


def run_agent(config=None) -> None:  # pragma: no cover - needs live infra
    """Run the LiveKit agent worker against LiveKit Cloud (or a self-hosted server).

    Connection comes from config: `LIVEKIT_URL` (wss://<project>.livekit.cloud), `LIVEKIT_API_KEY`,
    `LIVEKIT_API_SECRET`. Wiring (livekit-agents worker model):
      * the worker joins rooms created by the inbound SIP dispatch rule (voice/gateway/);
      * on participant audio: VAD -> STT -> Handler -> TTS -> publish audio;
      * VAD speech-start during playback -> session.barge_in() (interrupt TTS).
    STT/TTS come from config (STT_BACKEND/TTS_BACKEND) via build_session().
    """
    from common.config import DEFAULT  # noqa: PLC0415

    from .livekit_cloud import is_configured  # noqa: PLC0415

    config = config or DEFAULT
    if not is_configured(config):
        raise SystemExit(
            "LiveKit is not configured. Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET "
            "(LiveKit Cloud: wss://<project>.livekit.cloud) in your .env."
        )
    try:
        from livekit import agents  # noqa: F401, PLC0415
    except ImportError as exc:
        raise SystemExit(
            "livekit-agents not installed: `uv sync --extra livekit`, then re-run."
        ) from exc

    raise SystemExit(
        f"Connect the agent worker to {config.livekit_url} using livekit-agents and build_session(); "
        "see voice/gateway/README.md. cli/voice.py is the offline echo/orchestrator demo."
    )
