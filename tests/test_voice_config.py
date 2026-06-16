"""STT/TTS backend selection from config + clinical-vocab prompt wiring (no real models)."""

from __future__ import annotations

from dataclasses import replace

from common.config import CLINICAL_STT_PROMPT, DEFAULT
from voice.adapters import StubSTT, StubTTS, build_stt, build_tts


def test_default_backends_are_stub():
    assert isinstance(build_stt(), StubSTT)
    assert isinstance(build_tts(), StubTTS)


def test_clinical_prompt_default_has_arrhythmia_vocab():
    assert "atrial fibrillation" in CLINICAL_STT_PROMPT
    assert "acknowledge" in CLINICAL_STT_PROMPT
    assert DEFAULT.stt_initial_prompt == CLINICAL_STT_PROMPT


def test_build_stt_whisper_passes_model_and_prompt(monkeypatch):
    captured = {}

    class _FakeWhisper:
        def __init__(self, model, initial_prompt=None):
            captured["model"] = model
            captured["initial_prompt"] = initial_prompt

    monkeypatch.setattr("voice.adapters.WhisperSTT", _FakeWhisper)
    cfg = replace(DEFAULT, stt_backend="whisper", whisper_model="tiny.en")
    build_stt(cfg)
    assert captured["model"] == "tiny.en"
    assert "atrial fibrillation" in captured["initial_prompt"]  # G15 vocab biasing reached Whisper


def test_build_stt_respects_custom_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr("voice.adapters.WhisperSTT",
                        lambda model, initial_prompt=None: captured.update(p=initial_prompt))
    cfg = replace(DEFAULT, stt_backend="whisper", stt_initial_prompt="custom vocab here")
    build_stt(cfg)
    assert captured["p"] == "custom vocab here"
