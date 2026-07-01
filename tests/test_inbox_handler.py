"""Phase 9 Step 5: the in-app inbox chat handler.

Auth is by room membership (the app already PIN-authed at `/session`), so chat has no second PIN;
each turn is scoped to the selected worklist event, artifact-show requests fire an inline-render
callback, and chat declines until an event is selected. Collaborators are faked so this is offline
and deterministic (no LiveKit / graph / vector).
"""

from __future__ import annotations

import pytest

import kb.graph.events as events_mod
from common.audit import AuditLog
from orchestrator.orchestrator import TurnResult
from voice.handlers import InboxHandler, artifact_show_intent

ROOM = "rmsai-inbox-h1"


class _FakeOrch:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    def handle_turn(self, session_id: str, text: str) -> TurnResult:
        self.turns.append((session_id, text))
        if "weather" in text.lower():  # out of scope
            return TurnResult(answer="I can only help with this patient's clinical data.",
                              declined=True)
        return TurnResult(answer=f"Grounded: {text}")


class _FakeWorking:
    def __init__(self) -> None:
        self.scoped: dict[str, str | None] = {}

    def set_authenticated(self, session_id: str, *, patient_ref: str | None = None) -> None:
        self.scoped[session_id] = patient_ref


@pytest.fixture()
def handler(tmp_path, monkeypatch):
    monkeypatch.setattr(events_mod, "get_event_patient", lambda drv, uid: "PT1155")
    shown: list[tuple[str, str]] = []
    orch = _FakeOrch()
    working = _FakeWorking()
    h = InboxHandler(orch, working, driver=object(), on_show=lambda e, k: shown.append((e, k)),
                     audit=AuditLog(str(tmp_path / "audit.jsonl")))
    return h, orch, working, shown


# --- artifact-show intent (pure) ---------------------------------------------------------------

def test_artifact_show_intent():
    assert artifact_show_intent("show me the ECG strip") == "ecg_strip"
    assert artifact_show_intent("can you pull up the report") == "report"
    assert artifact_show_intent("let me see the heart rate trend") == "hr_trend"
    # a plain data question is NOT a show request
    assert artifact_show_intent("what was the heart rate?") is None
    assert artifact_show_intent("what were the vitals at the event?") is None


# --- handler behaviour -------------------------------------------------------------------------

def test_declines_until_an_event_is_selected(handler):
    h, orch, _working, _shown = handler
    reply = h.respond("what were the vitals?", session_id=ROOM)
    assert "Select an event" in reply
    assert orch.turns == []  # never touched the orchestrator without a scope


def test_selection_scopes_to_patient_then_answers(handler, tmp_path):
    h, orch, working, _shown = handler
    patient = h.set_selection(ROOM, "evt-1")
    assert patient == "PT1155"
    assert working.scoped[ROOM] == "PT1155"  # orchestrator turns are scoped to this patient

    reply = h.respond("what were the vitals at the event?", session_id=ROOM)
    assert reply == "Grounded: what were the vitals at the event?"
    assert orch.turns == [(ROOM, "what were the vitals at the event?")]
    line = h.audit.read_all()[-1]
    assert line["action"] == "phi_voice_query" and line["outcome"] == "answered"
    assert line["subject"] == "PT1155"


def test_select_control_message_scopes_without_a_reply(handler):
    h, orch, working, _shown = handler
    # The app scopes the conversation with a "/select <event_id>" control message on the chat
    # channel; it sets the scope and produces no chat bubble.
    reply = h.respond("/select evt-1", session_id=ROOM)
    assert reply == ""
    assert working.scoped[ROOM] == "PT1155"
    assert orch.turns == []  # control message never hits the orchestrator
    # a following question is now answered in-scope
    assert h.respond("what were the vitals?", session_id=ROOM).startswith("Grounded:")


def test_show_request_triggers_inline_render(handler):
    h, _orch, _working, shown = handler
    h.set_selection(ROOM, "evt-1")
    reply = h.respond("show me the ECG strip", session_id=ROOM)
    assert shown == [("evt-1", "ecg_strip")]   # app told to render it inline
    assert reply.startswith("Grounded:")        # ...and still gets a grounded answer


def test_plain_question_does_not_render(handler):
    h, _orch, _working, shown = handler
    h.set_selection(ROOM, "evt-1")
    h.respond("what was the heart rate?", session_id=ROOM)
    assert shown == []


def test_out_of_scope_declines(handler):
    h, _orch, _working, _shown = handler
    h.set_selection(ROOM, "evt-1")
    reply = h.respond("what's the weather tomorrow?", session_id=ROOM)
    assert "clinical" in reply.lower()
    assert h.audit.read_all()[-1]["outcome"] == "declined"


def test_selection_without_driver_falls_back_to_event_id(tmp_path):
    h = InboxHandler(_FakeOrch(), _FakeWorking(), driver=None,
                     audit=AuditLog(str(tmp_path / "a.jsonl")))
    assert h.set_selection(ROOM, "evt-xyz") == "evt-xyz"
