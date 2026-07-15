"""STT/TTS backend selection from config + clinical-vocab prompt wiring (no real models)."""

from __future__ import annotations

from dataclasses import replace

from common.config import CLINICAL_STT_PROMPT, DEFAULT
from voice.adapters import StubSTT, StubTTS, build_stt, build_tts, speakable


def test_default_backends_are_stub():
    assert isinstance(build_stt(), StubSTT)
    assert isinstance(build_tts(), StubTTS)


def test_speakable_strips_underscores_from_event_names():
    assert speakable("NORMAL_SINUS") == "NORMAL SINUS"           # spoken as words, no "underscore"
    assert speakable("AV_BLOCK_2_TYPE2") == "AV BLOCK 2 TYPE2"
    # a whole spoken line: only the machine token changes, the rest is untouched
    assert speakable("NORMAL_SINUS for PT8861, MEWS 0 (Low)") == "NORMAL SINUS for PT8861, MEWS 0 (Low)"


def test_speakable_is_a_noop_without_underscores():
    assert speakable("the heart rate is 92") == "the heart rate is 92"
    assert speakable("") == ""
    assert speakable(None) is None


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
