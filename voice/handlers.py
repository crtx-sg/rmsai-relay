"""Conversation handlers — text in, text out.

`EchoHandler` parrots the caller (Phase 5, to prove the audio loop). Phase 6 swaps in an
orchestrator-backed handler that authenticates the caller and returns grounded answers; both
satisfy the same `Handler` interface, so the voice session is unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Handler(ABC):
    @abstractmethod
    def respond(self, text: str, *, session_id: str) -> str: ...


class EchoHandler(Handler):
    """Returns what it heard (a parrot), proving STT -> handler -> TTS works end-to-end."""

    def respond(self, text: str, *, session_id: str) -> str:
        return text
