"""Offline tests for agent re-dispatch: the pure room-selection predicate + the cli.dispatch routing.

The LiveKit I/O (`create_agent_dispatch` / `redispatch_existing_rooms`) needs a live server and is
`pragma: no cover`; here we test the decision logic (`_should_dispatch`) and that the CLI routes to
the right call, both offline with the SDK faked out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli.dispatch as dispatch_cli
import voice.livekit_cloud as lkc
from voice.livekit_cloud import _AGENT_KIND, _has_fresh_active_job, _should_dispatch

HUMAN = 0          # ParticipantInfo.Kind.STANDARD
AGENT = _AGENT_KIND  # 4
PREFIX = "rmsai-inbox-"

JS_PENDING, JS_RUNNING, JS_SUCCESS, JS_FAILED = 0, 1, 2, 3
_S = 1_000_000_000            # ns per second
NOW = 2_000 * _S             # arbitrary "now" in unix ns
FRESH = NOW - 10 * _S        # started 10s ago (agent still cold-starting)
STALE = NOW - 600 * _S       # started 10 min ago (zombie RUNNING job from a dead worker)


def _job(status, started_ns):
    return SimpleNamespace(state=SimpleNamespace(status=status, started_at=started_ns))


def _dispatch(*jobs):
    """A fake AgentDispatch (duck-typed like the LiveKit proto) carrying the given jobs."""
    return SimpleNamespace(state=SimpleNamespace(jobs=list(jobs)))


# --- _should_dispatch (pure) -------------------------------------------------------------------

def test_dispatch_when_human_present_and_no_agent():
    assert _should_dispatch("rmsai-inbox-h1", [HUMAN], prefix=PREFIX, require_human=True) is True


def test_skip_when_agent_already_present():
    assert _should_dispatch("rmsai-inbox-h1", [HUMAN, AGENT], prefix=PREFIX,
                            require_human=True) is False


def test_skip_room_not_matching_prefix():
    assert _should_dispatch("rmsai-outbound-evt1", [HUMAN], prefix=PREFIX,
                            require_human=True) is False


def test_skip_empty_room_when_human_required():
    assert _should_dispatch("rmsai-inbox-h1", [], prefix=PREFIX, require_human=True) is False


def test_dispatch_empty_room_when_human_not_required():
    assert _should_dispatch("rmsai-inbox-h1", [], prefix=PREFIX, require_human=False) is True


def test_skip_room_in_cooldown_set():
    # A room we dispatched recently (still cold-starting) is skipped to avoid a second agent.
    assert _should_dispatch("rmsai-inbox-h1", [HUMAN], prefix=PREFIX, require_human=True,
                            skip={"rmsai-inbox-h1"}) is False


# --- _has_fresh_active_job (pure): the cold-start dedup that avoids duplicate agents (echo) ------

def test_fresh_running_job_blocks():
    # A just-dispatched agent (still cold-starting) blocks a second dispatch -> no echo.
    assert _has_fresh_active_job([_dispatch(_job(JS_RUNNING, FRESH))], now_ns=NOW) is True


def test_fresh_pending_job_blocks():
    assert _has_fresh_active_job([_dispatch(_job(JS_PENDING, FRESH))], now_ns=NOW) is True


def test_stale_running_job_does_not_block():
    # A zombie RUNNING job (dead/removed worker) must NOT block, or a restart never re-wires the room.
    assert _has_fresh_active_job([_dispatch(_job(JS_RUNNING, STALE))], now_ns=NOW) is False


def test_finished_job_never_blocks():
    assert _has_fresh_active_job(
        [_dispatch(_job(JS_SUCCESS, FRESH)), _dispatch(_job(JS_FAILED, FRESH))], now_ns=NOW) is False


def test_job_without_timestamp_treated_fresh():
    assert _has_fresh_active_job([_dispatch(_job(JS_RUNNING, 0))], now_ns=NOW) is True


def test_no_jobs_never_blocks():
    assert _has_fresh_active_job([], now_ns=NOW) is False
    assert _has_fresh_active_job([_dispatch()], now_ns=NOW) is False


def test_scans_all_dispatches_finds_the_fresh_one():
    assert _has_fresh_active_job(
        [_dispatch(_job(JS_RUNNING, STALE)), _dispatch(_job(JS_SUCCESS, FRESH), _job(JS_RUNNING, FRESH))],
        now_ns=NOW) is True


# --- cli.dispatch routing (LiveKit calls faked) ------------------------------------------------

def test_cli_room_routes_to_single_dispatch(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(lkc, "is_configured", lambda cfg: True)
    monkeypatch.setattr(dispatch_cli, "is_configured", lambda cfg: True)

    def _fake_create(room, *, config, **kw):
        calls["room"] = room
        return True

    monkeypatch.setattr(dispatch_cli, "create_agent_dispatch", _fake_create)
    rc = dispatch_cli.main(["--room", "rmsai-inbox-h1"])
    assert rc == 0
    assert calls["room"] == "rmsai-inbox-h1"
    assert "requested for room rmsai-inbox-h1" in capsys.readouterr().out


def test_cli_all_inbox_routes_to_redispatch(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(dispatch_cli, "is_configured", lambda cfg: True)

    def _fake_redispatch(config, *, require_human=True, **kw):
        seen["require_human"] = require_human
        return ["rmsai-inbox-h1", "rmsai-inbox-h2"]

    monkeypatch.setattr(dispatch_cli, "redispatch_existing_rooms", _fake_redispatch)
    rc = dispatch_cli.main(["--all-inbox"])
    assert rc == 0
    assert seen["require_human"] is True                    # skips empty rooms by default
    out = capsys.readouterr().out
    assert "rmsai-inbox-h1" in out and "2 room(s) dispatched" in out


def test_cli_all_inbox_include_empty_flips_require_human(monkeypatch):
    seen = {}
    monkeypatch.setattr(dispatch_cli, "is_configured", lambda cfg: True)

    def _fake_redispatch(config, *, require_human=True, **kw):
        seen["require_human"] = require_human
        return []

    monkeypatch.setattr(dispatch_cli, "redispatch_existing_rooms", _fake_redispatch)
    dispatch_cli.main(["--all-inbox", "--include-empty"])
    assert seen["require_human"] is False


def test_cli_requires_a_target(monkeypatch):
    monkeypatch.setattr(dispatch_cli, "is_configured", lambda cfg: True)
    with pytest.raises(SystemExit):  # mutually-exclusive group is required
        dispatch_cli.main([])
