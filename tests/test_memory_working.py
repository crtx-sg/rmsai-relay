"""Working memory: conversation state round-trips, incl. cross-instance (process) recall."""

from __future__ import annotations

import uuid

import pytest

from common.config import DEFAULT
from common.schemas import ChatTurn

pytestmark = pytest.mark.infra


def _redis_or_skip():
    redis = pytest.importorskip("redis")
    try:
        c = redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2)
        c.ping()
        return c
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"redis unreachable: {exc}")


def test_turn2_sees_turn1_across_instances():
    from memory.working import WorkingMemory

    client = _redis_or_skip()
    sid = f"sess-{uuid.uuid4().hex[:8]}"

    # "Process A" records turn 1
    a = WorkingMemory(client)
    a.append_turn(sid, ChatTurn(role="clinician", text="status of bed 3?", timestamp=1.0))

    # "Process B" — a fresh store instance — loads the session and sees turn 1
    b = WorkingMemory(client)
    state = b.load(sid)
    assert state is not None and len(state.turns) == 1
    assert state.turns[0].text == "status of bed 3?"

    # turn 2 depends on turn 1: appended on top of the loaded history
    b.append_turn(sid, ChatTurn(role="assistant", text="VT at 14:02", timestamp=2.0))
    reloaded = a.load(sid)
    assert [t.text for t in reloaded.turns] == ["status of bed 3?", "VT at 14:02"]

    a.clear(sid)
    assert a.load(sid) is None


def test_authentication_flag_persists():
    from memory.working import WorkingMemory

    client = _redis_or_skip()
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    wm = WorkingMemory(client)
    wm.set_authenticated(sid, patient_ref="PT1234")
    state = WorkingMemory(client).load(sid)
    assert state.authenticated and state.patient_ref == "PT1234"
    wm.clear(sid)
