"""Offline speech round-trip — prove the real local STT/TTS stack with no audio hardware.

Pipeline: text --Piper TTS--> WAV bytes --Whisper STT--> text. If what Whisper hears matches what
you typed, the self-hosted speech stack (the same `build_stt`/`build_tts` the LiveKit worker uses)
is working — independent of LiveKit, SIP, or a microphone. Handy on WSL2/headless boxes where
`cli.voice_worker console` can't open an audio device.

  uv run python -m cli.speech_check                       # one-shot, default clinical sentence
  uv run python -m cli.speech_check --text "bed four atrial fibrillation"
  uv run python -m cli.speech_check --interactive          # type lines, 'quit' to exit
  uv run python -m cli.speech_check --out /tmp/say.wav      # also save the WAV (play it on Windows)

Forces STT_BACKEND=whisper / TTS_BACKEND=piper (overriding .env) so it always tests the real
backends; model/voice paths still come from config (WHISPER_MODEL, PIPER_VOICE_PATH). Needs the
voice extra: `uv sync --extra voice`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from common.config import DEFAULT
from voice.adapters import STTAdapter, TTSAdapter, build_stt, build_tts

_DEFAULT_TEXT = "The patient in bed four has atrial fibrillation with a heart rate of 142."


def round_trip(text: str, *, stt: STTAdapter, tts: TTSAdapter) -> tuple[str, bytes]:
    """Synthesize `text` to audio, transcribe it back. Returns (heard, audio_bytes)."""
    audio = tts.synthesize(text)
    heard = stt.transcribe(audio)
    return heard, audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=_DEFAULT_TEXT, help="text to synthesize then transcribe")
    parser.add_argument("--interactive", action="store_true", help="loop, reading lines from stdin")
    parser.add_argument("--out", default=None, help="write the synthesized WAV to this path")
    args = parser.parse_args(argv)

    # Force the real self-hosted backends regardless of .env; keep model/voice paths from config.
    cfg = replace(DEFAULT, stt_backend="whisper", tts_backend="piper")
    print(f"loading Piper ({cfg.piper_voice_path}) + Whisper ({cfg.whisper_model})...")
    tts = build_tts(cfg)
    stt = build_stt(cfg)

    def _one(text: str) -> None:
        heard, audio = round_trip(text, stt=stt, tts=tts)
        match = "OK" if heard.strip().lower() == text.strip().lower() else "differs"
        print(f"  said  : {text!r}")
        print(f"  heard : {heard!r}  [{match}]")
        print(f"  audio : {len(audio)} bytes")
        if args.out:
            with open(args.out, "wb") as f:
                f.write(audio)
            print(f"  wrote : {args.out}")

    if args.interactive:
        print("type a line to synthesize+transcribe ('quit' to exit).")
        while True:
            try:
                line = input("say> ").strip()
            except EOFError:
                break
            if line.lower() in {"quit", "exit"}:
                break
            if line:
                _one(line)
    else:
        _one(args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
