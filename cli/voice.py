"""Phase 5/6 voice demo (terminal, no audio hardware).

Each line you type is "spoken" by the caller (encoded to stub audio), run through the same
VoiceSession the LiveKit agent uses, and the reply is "heard" back.

  python -m cli.voice                          # Phase 5: echo bot
  python -m cli.voice --barge-in-after 2       # interrupt playback after 2 words
  python -m cli.voice --mode orchestrator       # Phase 6: PIN-gated grounded answers (live backends)

In orchestrator mode, authenticate first by "saying" the PIN (default 1234, e.g. type
"one two three four"), then ask a clinical question.

The real audio path (LiveKit + Whisper + Piper + SIP) is in voice/livekit_agent.py + voice/gateway/.
"""

from __future__ import annotations

import argparse

from voice.adapters import StubSTT, StubTTS
from voice.handlers import build_handler
from voice.session import VoiceSession

_ECHO_BANNER = "voice echo demo — type what the caller says ('quit' to exit)."


def _build_handler(mode: str):
    """Return (handler, greeting, cleanup) for the terminal demo (shares voice.handlers factory)."""
    handler, greeting, cleanup = build_handler(mode)
    return handler, greeting or _ECHO_BANNER, cleanup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["echo", "orchestrator"], default="echo")
    parser.add_argument("--barge-in-after", type=int, default=None,
                        help="simulate the caller interrupting after N words (echo mode)")
    parser.add_argument("--session", default="cli-call")
    args = parser.parse_args(argv)

    handler, greeting, cleanup = _build_handler(args.mode)
    # Offline terminal demo: typed text is stub-encoded as caller "audio", so STT/TTS are stubs
    # here regardless of config. The real Whisper/Piper backends (config STT_BACKEND/TTS_BACKEND)
    # are wired in voice/livekit_agent.py via build_stt()/build_tts().
    session = VoiceSession(StubSTT(), StubTTS(), handler, args.session)
    print(greeting)
    try:
        while True:
            try:
                line = input("caller> ").strip()
            except EOFError:
                break
            if line.lower() in {"quit", "exit"}:
                break
            if not line:
                continue
            r = session.handle_turn(StubTTS().synthesize(line), barge_in_after=args.barge_in_after)
            heard_back = StubSTT().transcribe(r.audio)
            print(f"  bot heard : {r.heard!r}")
            print(f"  bot says  : {r.spoken!r}")
            print(f"  you hear  : {heard_back!r}" + ("  [barged in]" if r.interrupted else ""))
            print(f"  latency   : first-audio {r.metrics.first_audio_ms:.1f}ms, "
                  f"full-turn {r.metrics.full_turn_ms:.1f}ms")
    finally:
        if cleanup:
            cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
