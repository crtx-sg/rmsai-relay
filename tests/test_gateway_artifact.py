"""Phase 9 Step 4: the authenticated, audited artifact endpoint `GET /artifact/{token}`.

Bytes are released only behind a valid, single-event scoped token, and every hit is audited. ECG
strips and reports are served from disk; HR trend is a JSON series from the graph. Unknown / expired
tokens and missing files are refused (404) and audited, with no bytes.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from common.config import DEFAULT  # noqa: E402
from live import gateway  # noqa: E402
from live.artifact_tokens import ArtifactTokenStore  # noqa: E402
from live.gateway import create_app  # noqa: E402


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
    plots = tmp_path / "plots"
    reports = tmp_path / "reports"
    plots.mkdir()
    reports.mkdir()
    png = plots / "e1.png"
    png.write_bytes(b"\x89PNG\r\n_fake_png_bytes_")
    md = reports / "e1.md"
    md.write_text("# Event report\nAFib at bed ICU/3\n", encoding="utf-8")

    cfg = replace(DEFAULT, plot_dir=str(plots), report_dir=str(reports))
    info = {"patient": "PT1155", "ecg_plot_ref": str(png), "hr_history": [80, 95, 140],
            "hr_history_ts": [1.0, 2.0, 3.0], "report_uri": str(md)}
    monkeypatch.setattr(gateway, "get_event_artifacts",
                        lambda drv, uuid: None if uuid == "unknown" else info)

    redis = _FakeRedis()
    store = ArtifactTokenStore(redis, ttl_seconds=300)
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    client = TestClient(create_app(cfg, audit=audit, driver=object(), token_store=store))
    return {"client": client, "store": store, "audit": audit, "redis": redis, "png": png, "md": md}


def _last(audit):
    return audit.read_all()[-1]


def test_ecg_strip_served_and_audited(setup):
    token, _ = setup["store"].mint("e1", "ecg_strip")
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content == setup["png"].read_bytes()
    line = _last(setup["audit"])
    assert line["action"] == "view_artifact" and line["outcome"] == "served"
    assert line["subject"] == "PT1155" and line["extra"]["kind"] == "ecg_strip"


def test_hr_trend_served_as_json_series(setup):
    token, _ = setup["store"].mint("e1", "hr_trend")
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 200
    body = res.json()
    assert body == {"event_id": "e1", "hr_history": [80, 95, 140], "hr_history_ts": [1.0, 2.0, 3.0]}
    assert _last(setup["audit"])["extra"]["kind"] == "hr_trend"


def test_report_served_as_markdown(setup):
    token, _ = setup["store"].mint("e1", "report")
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/markdown")
    assert res.text == setup["md"].read_text()


def test_bad_token_is_404_audited_no_bytes(setup):
    res = setup["client"].get("/artifact/not-a-real-token")
    assert res.status_code == 404
    line = _last(setup["audit"])
    assert line["action"] == "view_artifact" and line["outcome"] == "denied"
    assert line["extra"]["reason"] == "bad_token"


def test_expired_token_is_404(setup):
    token, _ = setup["store"].mint("e1", "report")
    setup["redis"].clock += 301  # advance past TTL
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 404
    assert _last(setup["audit"])["outcome"] == "denied"


def test_unknown_event_is_404(setup):
    token, _ = setup["store"].mint("unknown", "report")
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 404
    assert _last(setup["audit"])["outcome"] == "unknown_event"


def test_missing_file_is_404_missing(setup):
    setup["png"].unlink()  # ref exists in the graph but the file is gone
    token, _ = setup["store"].mint("e1", "ecg_strip")
    res = setup["client"].get(f"/artifact/{token}")
    assert res.status_code == 404
    assert _last(setup["audit"])["outcome"] == "missing"


def test_token_cannot_be_replayed_for_another_kind(setup):
    # A report token must not fetch the ECG (kind is bound into the token).
    token, _ = setup["store"].mint("e1", "report")
    assert setup["store"].verify(token, "ecg_strip") is None
    # served correctly for its own kind
    assert setup["client"].get(f"/artifact/{token}").status_code == 200
    assert json.loads(json.dumps(_last(setup["audit"])))["extra"]["kind"] == "report"
