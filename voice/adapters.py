"""Speech adapters — STT and TTS behind small interfaces.

* `StubSTT` / `StubTTS` — deterministic, offline. They share a trivial codec (each audio "frame"
  is a word + NUL) so TTS output round-trips back through STT: this lets the echo loop be tested
  end-to-end (`speak X -> hear X`) with no audio hardware or models.
* `WhisperSTT` / `PiperTTS` — the real self-hosted backends (lazy import); used on real audio.

Audio is opaque `bytes` at this seam; only the adapters know the encoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from common.config import DEFAULT, Config

_SEP = b"\x00"


class STTAdapter(ABC):
    @abstractmethod
    def transcribe(self, audio: bytes) -> str: ...


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Yield audio in chunks (default: one chunk per word) so playback can be interrupted."""
        for word in text.split():
            yield word.encode("utf-8") + _SEP


class StubTTS(TTSAdapter):
    """Encodes text as recoverable 'audio' frames (one word per frame)."""

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))


class StubSTT(STTAdapter):
    """Decodes `StubTTS` audio (or caller audio produced the same way) back to text."""

    def transcribe(self, audio: bytes) -> str:
        words = [w.decode("utf-8") for w in audio.split(_SEP) if w]
        return " ".join(words)


class WhisperSTT(STTAdapter):
    """Self-hosted Whisper STT (lazy). Biased with clinical vocabulary (G15)."""

    def __init__(self, model: str = "base.en", initial_prompt: str | None = None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self._model = WhisperModel(model)
        self._initial_prompt = initial_prompt

    def transcribe(self, audio: bytes) -> str:
        import io  # noqa: PLC0415

        segments, _ = self._model.transcribe(io.BytesIO(audio), initial_prompt=self._initial_prompt)
        return " ".join(s.text.strip() for s in segments).strip()


class PiperTTS(TTSAdapter):
    """Self-hosted Piper TTS (lazy)."""

    def __init__(self, model_path: str) -> None:
        from piper.voice import PiperVoice  # noqa: PLC0415

        self._voice = PiperVoice.load(model_path)

    def synthesize(self, text: str) -> bytes:
        import io  # noqa: PLC0415

        buf = io.BytesIO()
        self._voice.synthesize(text, buf)
        return buf.getvalue()


def build_stt(config: Config = DEFAULT) -> STTAdapter:
    """Return the configured STT backend (stub by default; Whisper for real audio).

    Whisper is biased with the clinical vocabulary (`config.stt_initial_prompt`, G15) — this
    materially helps small models like `tiny.en` transcribe arrhythmia/drug/ack terms.
    """
    if config.stt_backend == "whisper":
        return WhisperSTT(model=config.whisper_model, initial_prompt=config.stt_initial_prompt)
    return StubSTT()


def build_tts(config: Config = DEFAULT) -> TTSAdapter:
    """Return the configured TTS backend (stub by default; Piper for real audio)."""
    if config.tts_backend == "piper":
        return PiperTTS(model_path=config.piper_voice_path)
    return StubTTS()
