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

    def __init__(self, model: str = "base.en", initial_prompt: str | None = None,
                 language: str | None = "en") -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self._model = WhisperModel(model)
        self._initial_prompt = initial_prompt
        # None / "" / "auto" -> let Whisper auto-detect; else pin the language (skips detection so it
        # never "hears" another language on noise). `.en` models are English-only regardless.
        self._language = language if language and language != "auto" else None

    def transcribe(self, audio: bytes) -> str:
        import io  # noqa: PLC0415

        segments, _ = self._model.transcribe(
            io.BytesIO(audio), language=self._language, initial_prompt=self._initial_prompt
        )
        return " ".join(s.text.strip() for s in segments).strip()


class PiperTTS(TTSAdapter):
    """Self-hosted Piper TTS (lazy). `sample_rate` comes from the loaded voice model."""

    def __init__(self, model_path: str) -> None:
        from piper.voice import PiperVoice  # noqa: PLC0415

        self._voice = PiperVoice.load(model_path)
        self.sample_rate: int = self._voice.config.sample_rate

    def synthesize(self, text: str) -> bytes:
        """Return a complete WAV (header + PCM) for `text`."""
        import io  # noqa: PLC0415
        import wave  # noqa: PLC0415

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:  # synthesize_wav sets channels/width/rate
            self._voice.synthesize_wav(text, wav)
        return buf.getvalue()

    def pcm_stream(self, text: str) -> Iterator[bytes]:
        """Yield raw little-endian int16 PCM (mono, `sample_rate`), one chunk per sentence.

        This is the low-latency primitive the LiveKit worker publishes frame-by-frame; the first
        chunk can play before the whole utterance is synthesized.
        """
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian int16 mono PCM in a WAV container."""
    import io  # noqa: PLC0415
    import wave  # noqa: PLC0415

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


# Fixed multipart boundary — fine for a single file part (collision with binary audio is negligible).
_BOUNDARY = "----rmsai-relay-boundary-7MA4YWxkTrZu0gW"


def _multipart(fields: dict[str, str], file_field: str, filename: str,
               file_bytes: bytes, file_ctype: str) -> tuple[bytes, str]:
    """Encode `fields` + one file part as multipart/form-data. Returns (body, content_type)."""
    crlf = b"\r\n"
    bnd = _BOUNDARY.encode()
    parts: list[bytes] = []
    for key, val in fields.items():
        parts += [b"--", bnd, crlf,
                  f'Content-Disposition: form-data; name="{key}"'.encode(), crlf, crlf,
                  str(val).encode(), crlf]
    parts += [b"--", bnd, crlf,
              f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode(),
              crlf, f"Content-Type: {file_ctype}".encode(), crlf, crlf, file_bytes, crlf,
              b"--", bnd, b"--", crlf]
    return b"".join(parts), f"multipart/form-data; boundary={_BOUNDARY}"


def _elevenlabs_request(req, *, timeout: int = 60) -> bytes:
    """Send an ElevenLabs request, raising a *legible* error (with the API's message) on failure.

    ElevenLabs returns the reason in a JSON body (e.g. 402 `paid_plan_required`: "Free users cannot
    use library voices…"); surfacing it beats a bare `HTTPError 402` traceback in the worker log.
    """
    import json  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            payload = json.loads(exc.read()).get("detail", {})
            detail = payload.get("message") or payload.get("code") or ""
        except Exception:  # noqa: BLE001 - body may be empty/non-JSON
            pass
        raise RuntimeError(f"ElevenLabs API error {exc.code}: {detail or exc.reason}") from exc


class ElevenLabsTTS(TTSAdapter):
    """Cloud TTS via ElevenLabs (stdlib HTTP). Requests raw PCM so the LiveKit worker can stream it.

    Cloud provider — for accuracy/latency benchmarking on SYNTHETIC data only (hard rules #4/#5).
    """

    _BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str, voice_id: str, model: str, sample_rate: int = 22050) -> None:
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set (required for TTS_BACKEND=elevenlabs)")
        self._key = api_key
        self._voice = voice_id
        self._model = model
        self.sample_rate = sample_rate

    def _request_pcm(self, text: str) -> bytes:
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        url = f"{self._BASE}/text-to-speech/{self._voice}?output_format=pcm_{self.sample_rate}"
        body = json.dumps({"text": text, "model_id": self._model}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "xi-api-key": self._key, "Content-Type": "application/json", "Accept": "audio/pcm",
        })
        return _elevenlabs_request(req)

    def synthesize(self, text: str) -> bytes:
        return _pcm_to_wav(self._request_pcm(text), self.sample_rate)

    def pcm_stream(self, text: str) -> Iterator[bytes]:
        yield self._request_pcm(text)


class ElevenLabsSTT(STTAdapter):
    """Cloud STT via ElevenLabs "Scribe" (stdlib HTTP). SYNTHETIC data only (hard rules #4/#5)."""

    _URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: str, model: str = "scribe_v1", language: str | None = "en") -> None:
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set (required for STT_BACKEND=elevenlabs)")
        self._key = api_key
        self._model = model
        # None / "" / "auto" -> let Scribe auto-detect; else pin it via `language_code`.
        self._language = language if language and language != "auto" else None

    def transcribe(self, audio: bytes) -> str:
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        fields = {"model_id": self._model}
        if self._language:
            fields["language_code"] = self._language
        body, ctype = _multipart(fields, "file", "audio.wav", audio, "audio/wav")
        req = urllib.request.Request(self._URL, data=body, headers={
            "xi-api-key": self._key, "Content-Type": ctype,
        })
        return json.loads(_elevenlabs_request(req)).get("text", "").strip()


class DeidentifyingTTS(TTSAdapter):
    """Scrub text through a `Deidentifier` before delegating to an inner (cloud) TTS.

    Defense-in-depth for cloud TTS: the pipeline already references patients by pseudonym (redaction
    by construction), but this guarantees that *every* spoken string — greeting, the event alert, and
    Q&A answers — is run through Presidio/regex before any text reaches a third party, so no name/
    SSN/etc. can leak. Fails closed (`DeidError`) — better silence than leaked PHI. `pcm_stream` and
    `sample_rate` are mirrored only when the inner adapter provides them (the worker probes for them).
    """

    def __init__(self, inner: TTSAdapter, deidentifier) -> None:
        self._inner = inner
        self._deid = deidentifier
        if hasattr(inner, "sample_rate"):
            self.sample_rate = inner.sample_rate
        if hasattr(inner, "pcm_stream"):
            self.pcm_stream = lambda text: inner.pcm_stream(self._clean(text))

    def _clean(self, text: str) -> str:
        from common.deid import deidentify  # noqa: PLC0415

        return deidentify(self._deid, text)

    def synthesize(self, text: str) -> bytes:
        return self._inner.synthesize(self._clean(text))


def build_stt(config: Config = DEFAULT) -> STTAdapter:
    """Return the configured STT backend (stub by default; Whisper local; ElevenLabs cloud).

    Whisper is biased with the clinical vocabulary (`config.stt_initial_prompt`, G15) — this
    materially helps small models like `tiny.en` transcribe arrhythmia/drug/ack terms.
    """
    if config.stt_backend == "whisper":
        return WhisperSTT(model=config.whisper_model, initial_prompt=config.stt_initial_prompt,
                          language=config.stt_language)
    if config.stt_backend == "elevenlabs":
        return ElevenLabsSTT(config.elevenlabs_api_key, config.elevenlabs_stt_model,
                             language=config.stt_language)
    return StubSTT()


def build_tts(config: Config = DEFAULT) -> TTSAdapter:
    """Return the configured TTS backend (stub by default; Piper local; ElevenLabs cloud)."""
    if config.tts_backend == "piper":
        return PiperTTS(model_path=config.piper_voice_path)
    if config.tts_backend == "elevenlabs":
        # Cloud TTS: wrap in a de-id pass so no PHI in any spoken text reaches the third party.
        from common.deid import get_deidentifier  # noqa: PLC0415

        inner = ElevenLabsTTS(config.elevenlabs_api_key, config.elevenlabs_voice_id,
                              config.elevenlabs_tts_model, config.elevenlabs_tts_sample_rate)
        return DeidentifyingTTS(inner, get_deidentifier(config.deid_backend))
    return StubTTS()
