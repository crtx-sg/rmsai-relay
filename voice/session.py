"""Voice session — turn-taking over the STT/handler/TTS adapters, with barge-in + latency.

A turn: caller audio -> STT -> handler -> TTS (streamed in chunks) -> played back. The caller can
**barge in** while the bot is speaking, which interrupts playback so the bot starts listening
again. Latency is measured as first-audio (time to the first synthesized chunk) and full-turn.

Audio I/O is opaque bytes here; the LiveKit agent (voice/livekit_agent.py) drives a real room with
the same session logic, wiring voice-activity detection to `barge_in`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .adapters import STTAdapter, TTSAdapter
from .handlers import Handler


@dataclass
class TurnMetrics:
    first_audio_ms: float
    full_turn_ms: float

    def within_budget(self, first_audio_ms: float | None, full_turn_ms: float | None) -> bool:
        ok = True
        if first_audio_ms is not None:
            ok = ok and self.first_audio_ms <= first_audio_ms
        if full_turn_ms is not None:
            ok = ok and self.full_turn_ms <= full_turn_ms
        return ok


@dataclass
class TurnResult:
    heard: str  # transcript of the caller
    spoken: str  # the handler's response text
    audio: bytes  # audio actually played back (partial if interrupted)
    interrupted: bool
    chunks_played: int
    metrics: TurnMetrics


class VoiceSession:
    def __init__(self, stt: STTAdapter, tts: TTSAdapter, handler: Handler, session_id: str) -> None:
        self.stt = stt
        self.tts = tts
        self.handler = handler
        self.session_id = session_id

    def handle_turn(self, caller_audio: bytes, *, barge_in_after: int | None = None) -> TurnResult:
        """Process one caller turn. `barge_in_after` (chunks) simulates the caller interrupting."""
        t0 = time.perf_counter()
        heard = self.stt.transcribe(caller_audio)
        spoken = self.handler.respond(heard, session_id=self.session_id)

        stream = self.tts.synthesize_stream(spoken)
        played: list[bytes] = []
        first_audio_ms = 0.0
        interrupted = False
        for i, chunk in enumerate(stream):
            if i == 0:
                first_audio_ms = (time.perf_counter() - t0) * 1000.0
            if barge_in_after is not None and i >= barge_in_after:
                interrupted = True  # caller spoke -> stop playing the rest
                break
            played.append(chunk)

        full_turn_ms = (time.perf_counter() - t0) * 1000.0
        return TurnResult(
            heard=heard, spoken=spoken, audio=b"".join(played),
            interrupted=interrupted, chunks_played=len(played),
            metrics=TurnMetrics(first_audio_ms=first_audio_ms, full_turn_ms=full_turn_ms),
        )
