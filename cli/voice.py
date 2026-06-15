"""Phase 5 voice echo-bot demo (terminal, no audio hardware).

Simulates a call: each line you type is "spoken" by the caller (encoded to stub audio), run
through the same VoiceSession the LiveKit agent uses, and the echoed reply is "heard" back.

  python -m cli.voice                         # interactive echo demo
  python -m cli.voice --barge-in-after 2      # interrupt playback after 2 words

The real audio path (LiveKit + Whisper + Piper + SIP) is in voice/livekit_agent.py +
voice/gateway/ and is verified manually with a phone.
"""

from __future__ import annotations

import argparse

from voice.adapters import StubSTT, StubTTS
from voice.handlers import EchoHandler
from voice.session import VoiceSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barge-in-after", type=int, default=None,
                        help="simulate the caller interrupting after N words")
    parser.add_argument("--session", default="cli-call")
    args = parser.parse_args(argv)

    tts_caller = StubTTS()  # turns your typed text into stub 'audio'
    session = VoiceSession(StubSTT(), StubTTS(), EchoHandler(), args.session)

    print("voice echo demo — type what the caller says ('quit' to exit).")
    while True:
        try:
            line = input("caller> ").strip()
        except EOFError:
            break
        if line.lower() in {"quit", "exit"}:
            break
        if not line:
            continue
        caller_audio = tts_caller.synthesize(line)
        r = session.handle_turn(caller_audio, barge_in_after=args.barge_in_after)
        heard_back = StubSTT().transcribe(r.audio)
        print(f"  bot heard : {r.heard!r}")
        print(f"  bot says  : {r.spoken!r}")
        print(f"  you hear  : {heard_back!r}" + ("  [barged in]" if r.interrupted else ""))
        print(f"  latency   : first-audio {r.metrics.first_audio_ms:.1f}ms, "
              f"full-turn {r.metrics.full_turn_ms:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
