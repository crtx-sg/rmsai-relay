"""Probe the inbox room's chat/push-to-talk control path WITHOUT the browser.

Joins `rmsai-inbox-<hospital_id>` as a clinician participant, sends messages on the `lk.chat` topic
exactly as the companion app does, and prints whatever the agent sends back. This isolates the two
halves of an "in-app chat/PTT doesn't work" report: if the probe gets replies, the worker is healthy
and the fault is in the browser client (stale `app.js`, a handler that never fires, a failed send);
if it doesn't, the fault is in the worker and the browser is a red herring.

  # push-to-talk control frames (no audio published, so the agent should answer "I didn't catch that")
  uv run python -m cli.inbox_probe --ptt

  # a typed question, scoped to a worklist event
  uv run python -m cli.inbox_probe --select <event-uuid> --say "what were the vitals at the event?"

  # did selecting a row actually speak the report? (🔊 lines are what the agent said aloud)
  uv run python -m cli.inbox_probe --select <event-uuid>

Needs LiveKit configured (`LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`) and the voice worker
running. NOTE: joining makes this probe a room participant, so the agent may re-link its audio input
here — run it when the app isn't mid-test, or reload the app afterwards.
"""

from __future__ import annotations

import argparse
import asyncio

from common.config import DEFAULT, Config
from live.inbox import inbox_room
from voice.livekit_cloud import access_token, is_configured

_TOPIC = "lk.chat"
#: The agent mirrors everything it *speaks* here (livekit.agents TOPIC_TRANSCRIPTION). Listening to
#: it turns "did the TTS actually run?" into something you can see without audio hardware — which is
#: the only way to tell a silent speaker apart from a speech step that never fired.
_TOPIC_SPOKEN = "lk.transcription"


async def _run(messages: list[str], *, wait_s: float, config: Config) -> int:
    from livekit import rtc  # noqa: PLC0415

    room_name = inbox_room(config)
    identity = "probe-cli"
    token = access_token(identity=identity, room=room_name, name="probe", config=config)

    room = rtc.Room()
    replies: list[str] = []
    spoken: list[str] = []

    def _reader_for(sink: list[str], arrow: str):
        def _handler(reader, participant_identity: str) -> None:
            async def _read() -> None:
                text = await reader.read_all()
                sink.append(text)
                print(f"  {arrow} {participant_identity}: {text}", flush=True)

            asyncio.create_task(_read())  # noqa: RUF006 - fire-and-forget print task

        return _handler

    room.register_text_stream_handler(_TOPIC, _reader_for(replies, "<-"))
    room.register_text_stream_handler(_TOPIC_SPOKEN, _reader_for(spoken, "🔊"))
    await room.connect(config.livekit_url, token)
    print(f"joined {room_name!r} as {identity}", flush=True)
    agents = [p.identity for p in room.remote_participants.values()
              if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]
    print(f"agents in room: {agents or 'NONE (no worker joined — nothing will answer)'}", flush=True)

    try:
        for msg in messages:
            print(f"  -> {msg}", flush=True)
            await room.local_participant.send_text(msg, topic=_TOPIC)
            await asyncio.sleep(1.0)
        print(f"waiting {wait_s:.0f}s for replies…", flush=True)
        await asyncio.sleep(wait_s)
    finally:
        await room.disconnect()

    print(f"\n{len(replies)} typed repl{'y' if len(replies) == 1 else 'ies'}, "
          f"{len(spoken)} spoken segment{'' if len(spoken) == 1 else 's'}", flush=True)
    return 0 if (replies or spoken) else 1


def build_messages(*, select: str | None, say: list[str], ptt: bool) -> list[str]:
    """Assemble the message sequence to send, in the order the companion app would send it.

    A selection has to precede the question (chat is scoped to a worklist row), and the audio turn
    brackets whatever is said inside it — `/ptt-start`, speech, `/ptt-end`.
    """
    messages: list[str] = []
    if select:
        messages.append(f"/select {select}")
    if ptt:
        messages.append("/ptt-start")
    messages.extend(say)
    if ptt:
        messages.append("/ptt-end")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", help="Send '/select <event-uuid>' first (scopes the chat).")
    parser.add_argument("--say", action="append", default=[], help="Message to send (repeatable).")
    parser.add_argument("--ptt", action="store_true",
                        help="Send the push-to-talk control frames (/ptt-start, /ptt-end).")
    parser.add_argument("--wait", type=float, default=25.0, help="Seconds to wait for replies.")
    args = parser.parse_args(argv)

    if not is_configured(DEFAULT):
        parser.error("LiveKit is not configured (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET)")

    messages = build_messages(select=args.select, say=args.say, ptt=args.ptt)
    if not messages:
        parser.error("nothing to send: pass --ptt, --say and/or --select")

    return asyncio.run(_run(messages, wait_s=args.wait, config=DEFAULT))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
