"""Live call 2026-11-04 (h1, PT985263): "Okay, acknowledge this statement" (STT for "event") was not
an acknowledgment and went to Q&A, which read out a ward-wide alarm list; then a DNS blip killed
the ElevenLabs STT, the agent session closed on the error, and the event stayed answered but open.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from voice.livekit_agent import redial_on_session_error
from voice.outbound import parse_ack


def test_statement_is_stt_for_event():
    assert parse_ack("Okay, acknowledge this statement") == "yes"


def test_an_unparsed_ack_command_confirms_back_instead_of_answering(tmp_path):
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking, _handler

    working, orch = _FakeWorking(), _FakeOrch()
    h, _ = _handler(tmp_path, working, orch, require_pin=False)
    h.respond("yes", session_id="room1")
    reply = h.respond("acknowledge it now please doctor", session_id="room1")
    assert reply.startswith("Do you want to acknowledge") and orch.calls == []  # no Q&A
    assert not h.is_ended("room1")  # nothing closed yet
    assert h.respond("acknowledge", session_id="room1").endswith("Goodbye.")


class _Session:
    def __init__(self):
        self.handlers = {}

    def on(self, event, fn):
        self.handlers[event] = fn


class _Handler:
    def __init__(self, *, authenticated=True, ended=False):
        self.alert = SimpleNamespace(event_id="E1")
        self._auth, self._ended = authenticated, ended

    def is_authenticated(self, _room):
        return self._auth

    def is_ended(self, _room):
        return self._ended


def _close(handler, reason):
    session, marked, deleted = _Session(), [], []

    async def _delete():
        deleted.append(True)

    async def _run():
        redial_on_session_error(session, handler, "room1", mark_dropped=marked.append,
                                delete_room=_delete)
        session.handlers["close"](SimpleNamespace(reason=reason, error=RuntimeError("dns")))
        await asyncio.sleep(0)

    asyncio.run(_run())
    return marked, deleted


def test_a_failed_session_hangs_up_and_rings_again():
    assert _close(_Handler(), "error") == (["E1"], [True])


def test_no_redial_once_acknowledged_or_on_a_normal_close():
    assert _close(_Handler(ended=True), "error") == ([], [])
    assert _close(_Handler(), "participant_disconnected") == ([], [])
    # failed before the "yes": a missed attempt for the retry policy, no extra redial flag
    assert _close(_Handler(authenticated=False), "error") == ([], [True])


def _reconnect(*, answered=True, ended=False):
    from voice.livekit_agent import end_call_on_reconnect

    room, marked, deleted = _Session(), [], []
    room.name = "room1"

    async def _delete():
        deleted.append(True)

    async def _run():
        end_call_on_reconnect(room, "E1", mark_dropped=marked.append, delete_room=_delete,
                              answered=lambda: answered, ended=lambda: ended)
        room.handlers["reconnecting"]()
        flagged_before_reconnect = list(marked)
        room.handlers["reconnected"]()
        await asyncio.sleep(0)
        return flagged_before_reconnect

    return asyncio.run(_run()), marked, deleted


def test_a_dropped_call_is_flagged_before_the_phone_leg_ends():
    # live 2026-11-04 (PT914543): the call ended 6 s before LiveKit reconnected; flag was too late
    assert _reconnect() == (["E1"], ["E1"], [True])  # flagged once, then hung up


def test_no_redial_after_an_acknowledged_call_drops():
    assert _reconnect(ended=True) == ([], [], [])
