"""ElevenLabs cloud STT/TTS adapters — offline tests with mocked HTTP (no network, no API key).

Verifies request shape (URL/headers/body), response parsing, WAV wrapping, and build_* dispatch.
"""

from __future__ import annotations

import io
import json
import wave
from dataclasses import replace

import pytest

from common.config import DEFAULT
from voice.adapters import (
    DeidentifyingTTS,
    ElevenLabsSTT,
    ElevenLabsTTS,
    TTSAdapter,
    build_stt,
    build_tts,
)


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, payload: bytes):
    """Capture the urllib Request and return `payload`. Returns a dict the test can inspect."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = req.data
        return _FakeResp(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_tts_requests_pcm_and_wraps_wav(monkeypatch):
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"  # 4 samples of int16 PCM
    cap = _patch_urlopen(monkeypatch, pcm)
    tts = ElevenLabsTTS("KEY", "voice123", "eleven_flash_v2_5", sample_rate=22050)

    wav = tts.synthesize("hello")
    # request shape
    assert "text-to-speech/voice123" in cap["url"]
    assert "output_format=pcm_22050" in cap["url"]
    assert cap["headers"]["xi-api-key"] == "KEY"
    assert json.loads(cap["body"]) == {"text": "hello", "model_id": "eleven_flash_v2_5"}
    # WAV wrapping: parseable, mono/16-bit/22050, PCM preserved
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 22050)
        assert w.readframes(w.getnframes()) == pcm


def test_tts_pcm_stream_yields_raw_pcm(monkeypatch):
    pcm = b"\x10\x00\x20\x00"
    _patch_urlopen(monkeypatch, pcm)
    tts = ElevenLabsTTS("KEY", "v", "m", sample_rate=16000)
    assert list(tts.pcm_stream("hi")) == [pcm]


def test_stt_posts_multipart_and_parses_text(monkeypatch):
    cap = _patch_urlopen(monkeypatch, json.dumps({"text": "  one two three four  "}).encode())
    stt = ElevenLabsSTT("KEY", "scribe_v1")

    text = stt.transcribe(b"RIFFfake-wav-bytes")
    assert text == "one two three four"  # stripped
    assert cap["url"].endswith("/v1/speech-to-text")
    assert cap["headers"]["xi-api-key"] == "KEY"
    assert cap["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    assert b'name="model_id"' in cap["body"] and b"scribe_v1" in cap["body"]
    assert b'filename="audio.wav"' in cap["body"] and b"RIFFfake-wav-bytes" in cap["body"]


def test_stt_forces_language_code(monkeypatch):
    # default "en" -> language_code is sent; "auto" -> not sent (Scribe auto-detects).
    cap = _patch_urlopen(monkeypatch, json.dumps({"text": "hi"}).encode())
    ElevenLabsSTT("KEY", "scribe_v1", language="en").transcribe(b"wav")
    assert b'name="language_code"' in cap["body"] and b"\r\nen\r\n" in cap["body"]

    cap2 = _patch_urlopen(monkeypatch, json.dumps({"text": "hi"}).encode())
    ElevenLabsSTT("KEY", "scribe_v1", language="auto").transcribe(b"wav")
    assert b'name="language_code"' not in cap2["body"]


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        ElevenLabsTTS("", "v", "m")
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        ElevenLabsSTT("")


def test_build_dispatch_to_elevenlabs():
    cfg = replace(DEFAULT, stt_backend="elevenlabs", tts_backend="elevenlabs",
                  elevenlabs_api_key="KEY")
    assert isinstance(build_stt(cfg), ElevenLabsSTT)
    # Cloud TTS is wrapped in the de-id guard, not returned bare.
    tts = build_tts(cfg)
    assert isinstance(tts, DeidentifyingTTS)


# --- de-id-before-cloud-TTS guard ---


class _RecordingTTS(TTSAdapter):
    """Inner TTS that records the (already de-identified) text it was asked to synthesize."""

    sample_rate = 22050

    def __init__(self) -> None:
        self.seen: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.seen.append(text)
        return b"WAV" + text.encode()

    def pcm_stream(self, text: str):
        self.seen.append(text)
        yield b"PCM" + text.encode()


class _FakeDeid:
    """Stand-in Deidentifier: redacts the literal 'Jane Doe' so we can assert the scrub ran."""

    def deidentify(self, text: str) -> str:
        return text.replace("Jane Doe", "[REDACTED]")


def test_deid_tts_scrubs_before_synthesize():
    inner = _RecordingTTS()
    tts = DeidentifyingTTS(inner, _FakeDeid())
    tts.synthesize("Alert for Jane Doe in bed 4")
    assert inner.seen == ["Alert for [REDACTED] in bed 4"]  # inner never saw the name


def test_deid_tts_scrubs_pcm_stream_and_mirrors_sample_rate():
    inner = _RecordingTTS()
    tts = DeidentifyingTTS(inner, _FakeDeid())
    assert tts.sample_rate == 22050               # mirrored from inner (worker reads it)
    list(tts.pcm_stream("hi Jane Doe"))
    assert inner.seen == ["hi [REDACTED]"]


def test_deid_tts_hides_pcm_stream_when_inner_lacks_it():
    class _NoStream(TTSAdapter):
        def synthesize(self, text: str) -> bytes:
            return b""

    tts = DeidentifyingTTS(_NoStream(), _FakeDeid())
    assert not hasattr(tts, "pcm_stream")         # LocalTTS probes getattr -> falls back to WAV
    assert not hasattr(tts, "sample_rate")
