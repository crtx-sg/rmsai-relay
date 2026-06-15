"""Working memory — conversation state persisted in Redis.

Stores/loads `ConversationState` by `session_id`, so a session resumes across turns and across
processes. This is the checkpointer primitive the Phase 4 LangGraph orchestrator builds on
(LangGraph's Redis saver can replace it without changing callers).
"""

from __future__ import annotations

from common.config import DEFAULT, Config
from common.schemas import ChatTurn, ConversationState

_KEY = "rmsai:session:{sid}"


class WorkingMemory:
    def __init__(self, redis_client, ttl_seconds: int | None = None) -> None:
        self.redis = redis_client
        self.ttl = ttl_seconds

    @classmethod
    def from_config(cls, config: Config = DEFAULT, ttl_seconds: int | None = None) -> "WorkingMemory":
        import redis  # noqa: PLC0415

        return cls(redis.Redis.from_url(config.redis_url), ttl_seconds)

    def save(self, state: ConversationState) -> None:
        key = _KEY.format(sid=state.session_id)
        self.redis.set(key, state.model_dump_json(), ex=self.ttl)

    def load(self, session_id: str) -> ConversationState | None:
        raw = self.redis.get(_KEY.format(sid=session_id))
        if raw is None:
            return None
        return ConversationState.model_validate_json(raw)

    def get_or_create(self, session_id: str) -> ConversationState:
        return self.load(session_id) or ConversationState(session_id=session_id)

    def append_turn(self, session_id: str, turn: ChatTurn) -> ConversationState:
        """Load, append a turn, persist, and return the updated state (so turn N sees 1..N-1)."""
        state = self.get_or_create(session_id)
        state.turns.append(turn)
        self.save(state)
        return state

    def set_authenticated(self, session_id: str, *, patient_ref: str | None = None) -> None:
        state = self.get_or_create(session_id)
        state.authenticated = True
        if patient_ref is not None:
            state.patient_ref = patient_ref
        self.save(state)

    def clear(self, session_id: str) -> None:
        self.redis.delete(_KEY.format(sid=session_id))
