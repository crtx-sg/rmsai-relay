"""Run the live LiveKit agent worker (real audio: phone/web <-> Whisper/Piper <-> Handler).

Unlike `cli.voice` (offline, typed-text demo), this connects to a real LiveKit endpoint and drives
actual audio. rmsai-relay dials the clinician into a room (voice/outbound.py); this worker joins it.

  uv run python -m cli.voice_worker dev      # autoreload dev worker
  uv run python -m cli.voice_worker start     # production worker
  VOICE_MODE=echo uv run python -m cli.voice_worker dev   # parrot handler (loopback test)

Requires: `uv sync --extra livekit --extra voice`, a LiveKit endpoint + key/secret in .env
(LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET), and STT_BACKEND=whisper / TTS_BACKEND=piper.
The subcommand (dev/start/connect/...) is parsed by livekit-agents' own CLI.
"""

from __future__ import annotations

from voice.livekit_agent import run_agent

if __name__ == "__main__":
    run_agent()
