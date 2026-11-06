"""Live end-to-end call 2026-11-06 (h1 + h2): fixes for what the transcripts showed."""

from __future__ import annotations

import time

from common.schemas import ChatTurn
from tests.test_episodic_gate import _orch, _SpyEpisodic

_ALERT = "Alert for patient PT970326 on bed 21. Detected sinus bradycardia."
_DECLINE = "I don't have information on that in the knowledge base."


def _seeded():
    orch = _orch(_SpyEpisodic(), episodic_recall=False)
    turns = orch.working.get_or_create("s").turns
    turns += [ChatTurn(role="assistant", text=_ALERT, timestamp=time.time(), mode="alert"),
              ChatTurn(role="assistant", text=_DECLINE, timestamp=time.time())]
    return orch


def test_repeat_skips_a_decline_and_revoices_the_alert():
    orch = _seeded()
    assert orch.handle_turn("s", "Repeat", spoken=True).answer == _ALERT
    assert orch.handle_turn("s", "repeat", spoken=True).answer == _ALERT  # not the repeat's echo


def test_pardon_on_a_call_is_a_repeat():
    for said in ("Eh?", "sorry?", "Pardon me?", "Hello?", "what?"):
        r = _seeded().handle_turn("s", said, spoken=True)
        assert (r.mode, r.answer) == ("repeat", _ALERT), said
    assert _seeded().handle_turn("s", "Hello?").mode != "repeat"  # typed chat: unchanged
