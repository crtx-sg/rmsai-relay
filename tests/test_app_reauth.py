"""Companion app session expiry: a 401 from the gateway prompts for the PIN once, then retries.

The session token is the LiveKit join token (1 h). The room connection outlives it, so after an idle
hour the worklist and audio still work while artifact / event-info / ack / metrics calls get 401.
Drives `app/app.js` against a fake gateway in node (tests/js/reauth_smoke.js). Skipped without node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def run():
    out = subprocess.run(["node", str(_ROOT / "tests/js/reauth_smoke.js"), str(_ROOT / "app/app.js")],
                         capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def test_concurrent_401s_share_one_pin_prompt(run):
    assert run["promptsShown"] == 1


def test_wrong_pin_keeps_the_prompt_open(run):
    assert run["wrongPinMessage"] == "Incorrect PIN." and run["openAfterWrongPin"] is True


def test_right_pin_swaps_the_token_and_retries_each_call(run):
    assert run["statuses"] == [200, 200]
    assert run["sessionCalls"] == 2  # the wrong PIN, then the right one
    assert run["retriedWithNewToken"] == ["old", "old", "new", "new"]
    assert run["tokenNow"] == "new"


def test_cancel_returns_the_401_without_signing_in(run):
    assert run["cancelStatus"] == 401 and run["sessionCallsOnCancel"] == 0
