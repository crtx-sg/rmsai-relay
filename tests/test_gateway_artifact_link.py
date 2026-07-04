"""Phase 9: on-demand artifact links via `POST /artifact-link`.

Worklist links carried in an inbox `event` message are minted at publish time and expire fast, so a
link clicked minutes later 404s. The app instead mints a FRESH scoped token at click time behind the
same session-token gate as `/ack`. This test proves: a valid session mints a token that then resolves
via `GET /artifact/{token}`; a bad/wrong-room session is refused (401); an unknown kind is refused
(400); and the whole thing is fail-closed + audited.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from common.config import DEFAULT  # noqa: E402
from live import gateway  # noqa: E402
from live.artifact_tokens import ArtifactTokenStore  # noqa: E402
from live.gateway import create_app  # noqa: E402
from voice.livekit_cloud import access_token  # noqa: E402

_CFG = replace(
    DEFAULT, hospital_id="h1", inbound_auth_pin="1234",
    livekit_url="ws://lk:7880", livekit_api_key="devkey", livekit_api_secret="devsecret",
)


class _FakeRedis:
    def __init__(self, clock: float = 1000.0) -> None:
        self._d: dict[str, tuple[str, float | None]] = {}
        self.clock = clock

    def set(self, key, val, ex=None):
        self._d[key] = (val, None if ex is None else self.clock + ex)

    def get(self, key):
        item = self._d.get(key)
        if item is None:
            return None
        val, exp = item
        if exp is not None and self.clock >= exp:
            self._d.pop(key, None)
            return None
        return val.encode("utf-8")

    def delete(self, key):
        self._d.pop(key, None)


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    # A report artifact the minted token can resolve to at serve time.
    reports = tmp_path / "reports"
    reports.mkdir()
    md = reports / "e1.md"
    md.write_text("# Event report\nAFib\n", encoding="utf-8")
    cfg = replace(_CFG, report_dir=str(reports))
    info = {"patient": "PT1155", "report_uri": str(md)}
    monkeypatch.setattr(gateway, "get_event_artifacts",
                        lambda drv, uuid: None if uuid == "unknown" else info)

    store = ArtifactTokenStore(_FakeRedis(), ttl_seconds=300)
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    client = TestClient(create_app(cfg, audit=audit, driver=object(), token_store=store))
    return {"client": client, "store": store, "audit": audit}


def _token(room="rmsai-inbox-h1"):
    return access_token(identity="clinician-x", room=room, config=_CFG)


def _last(audit):
    return audit.read_all()[-1]


def test_mints_fresh_link_that_resolves(setup):
    res = setup["client"].post(
        "/artifact-link", json={"event_id": "e1", "kind": "report", "session": _token()}
    )
    assert res.status_code == 200
    url = res.json()["url"]
    assert url.startswith("/artifact/")
    assert _last(setup["audit"])["action"] == "mint_artifact_link"
    assert _last(setup["audit"])["outcome"] == "minted"

    # The freshly minted token actually serves the bytes.
    served = setup["client"].get(url)
    assert served.status_code == 200
    assert served.text.startswith("# Event report")


def test_bad_session_is_refused(setup):
    res = setup["client"].post(
        "/artifact-link", json={"event_id": "e1", "kind": "report", "session": "not.a.token"}
    )
    assert res.status_code == 401
    assert _last(setup["audit"])["outcome"] == "unauthorized"


def test_wrong_room_session_is_refused(setup):
    res = setup["client"].post(
        "/artifact-link",
        json={"event_id": "e1", "kind": "report", "session": _token("rmsai-inbox-other")},
    )
    assert res.status_code == 401


def test_unknown_kind_is_refused(setup):
    res = setup["client"].post(
        "/artifact-link", json={"event_id": "e1", "kind": "not-a-kind", "session": _token()}
    )
    assert res.status_code == 400


def test_each_call_mints_a_distinct_token(setup):
    body = {"event_id": "e1", "kind": "report", "session": _token()}
    a = setup["client"].post("/artifact-link", json=body).json()["url"]
    b = setup["client"].post("/artifact-link", json=body).json()["url"]
    assert a != b  # fresh token each click, not a reused publish-time link
