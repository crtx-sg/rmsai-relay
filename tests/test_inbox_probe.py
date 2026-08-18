"""Inbox probe CLI — message assembly (the live LiveKit join needs infra and isn't unit-tested).

The probe exists to split "in-app chat doesn't work" into worker-side vs browser-side: it speaks the
same `lk.chat` control protocol the companion app does, so a reply proves the worker half is healthy.
That protocol is what these tests pin — the order matters (scope before question, audio turn brackets
the speech), and a wrong order would make the probe lie about which half is broken.
"""

from __future__ import annotations

from cli.inbox_probe import build_messages


def test_ptt_frames_bracket_the_turn():
    assert build_messages(select=None, say=[], ptt=True) == ["/ptt-start", "/ptt-end"]
    assert build_messages(select=None, say=["hello"], ptt=True) == [
        "/ptt-start", "hello", "/ptt-end"]


def test_selection_precedes_everything():
    # Chat is scoped to a worklist row: a question sent before the selection gets the
    # "select an event first" decline instead of an answer.
    assert build_messages(select="evt-1", say=["what were the vitals?"], ptt=False) == [
        "/select evt-1", "what were the vitals?"]
    assert build_messages(select="evt-1", say=[], ptt=True)[0] == "/select evt-1"


def test_nothing_requested_is_empty():
    assert build_messages(select=None, say=[], ptt=False) == []
