"""`POST /events` (Events tab + worklist backfill) and the chat → Events-tab link."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from live import gateway  # noqa: E402
from live.gateway import create_app  # noqa: E402
from orchestrator.orchestrator import history_filter_for  # noqa: E402
from orchestrator.perf_view import build_events_view  # noqa: E402
from tests.test_gateway_perf import _CFG, _row, _token  # noqa: E402
from voice.handlers import InboxHandler  # noqa: E402

VT, NSR = "VENTRICULAR_TACHYCARDIA", "NORMAL_SINUS"


def _hist(eid, pred, truth=None, *, pending=True, **kw):
    r = _row(eid, pred, truth, **kw)
    r.update(bed="Unit1-Bed02", unit="Unit1", status="reported", pending=pending,
             artifact_kinds=["ecg_strip", "report"],
             alert_reason="vitals_alert (MEWS 6 >= threshold 3)", alert_basis="vitals")
    return r


@pytest.fixture()
def graph(monkeypatch):
    seen = {}

    def fake_list(driver, **kw):
        seen["filters"] = kw
        rows = [_hist("e1", VT, VT), _hist("e2", NSR, NSR, pending=False, gate=False,
                                           eval_outcome="TN")]
        return rows[: kw.get("limit") or None]

    monkeypatch.setattr(gateway, "list_events", fake_list)
    return seen


def _client(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    return TestClient(create_app(_CFG, audit=audit, driver=object())), audit


def test_events_lists_with_the_filters_and_counts(tmp_path, graph):
    client, audit = _client(tmp_path)
    res = client.post("/events", json={"session": _token(), "status": "pending", "bed": "bed 2",
                                       "event_type": VT, "since": "24h"})
    assert res.status_code == 200
    b = res.json()
    assert (b["total"], b["pending"], b["truncated"]) == (2, 1, False)
    assert b["filters"] == {"status": "pending", "bed": "bed 2", "event_type": VT, "since": "24h"}
    f = graph["filters"]
    assert (f["status"], f["bed"], f["event_type"], f["labelled_only"]) == ("pending", "bed 2", VT,
                                                                           False)
    assert f["since"] is not None and f["limit"] == 501
    assert "models" not in b  # the performance summary only in labelled mode
    rec = audit.read_all()[-1]
    assert rec["action"] == "view_events" and rec["extra"]["filters"] == "bed,event_type,since,status"


def test_rows_render_like_a_live_worklist_message(tmp_path, graph):
    client, _ = _client(tmp_path)
    e = client.post("/events", json={"session": _token()}).json()["events"][0]
    # the worklist backfill applies these as `type:"event"` messages: same keys as a live push
    for key in ("event_id", "patient", "bed", "unit", "event_type", "ts", "criticality",
                "status", "confidence", "low_confidence", "alert_basis", "alert_reason", "why",
                "source", "truth", "outcome", "links"):
        assert key in e
    assert e["links"] == {"ecg_strip": {}, "report": {}}  # kinds only; URLs minted on click
    assert e["alert_reason"] == "MEWS 6 >= threshold 3" and e["low_confidence"] is True
    assert e["pending"] is True and e["alert"] is True
    assert e["source"] == "INCART I05 @263874 (test split)"  # demo data keeps its badge


def test_labelled_mode_adds_the_model_summary(tmp_path, graph):
    client, _ = _client(tmp_path)
    b = client.post("/events", json={"session": _token(), "labelled": True}).json()
    assert graph["filters"]["labelled_only"] is True
    assert [m["model_id"] for m in b["models"]] == ["ecg_transconv:v2"]
    assert b["models"][0]["summary"]["counts"]["TP"] == 1


def test_truncation_is_reported(tmp_path, graph):
    client, _ = _client(tmp_path)
    b = client.post("/events", json={"session": _token(), "limit": 1}).json()
    assert b["total"] == 1 and b["truncated"] is True


def test_events_requires_a_session_and_valid_filters(tmp_path, graph):
    client, audit = _client(tmp_path)
    assert client.post("/events", json={"session": "bad"}).status_code == 401
    assert "filters" not in graph and audit.read_all()[-1]["outcome"] == "unauthorized"
    assert client.post("/events", json={"session": _token(), "since": "soon"}).status_code == 400


def test_real_device_rows_carry_no_source_badge():
    row = _hist("e1", VT, source_kind="device", source_device="RMS-1")
    assert build_events_view([row])[0]["source"] is None


def test_history_filter_reproduces_the_listing():
    assert history_filter_for({"status": "pending", "bed": "Unit1-Bed02", "patient_id": "PT1",
                               "event_type": VT, "min_criticality": "High", "since": 1000.0,
                               "window_hours": 6}) == {
        "status": "pending", "bed": "Unit1-Bed02", "patient": "PT1", "event_type": VT,
        "min_criticality": "High", "since": "1000.0"}


class _Orch:
    def __init__(self, result):
        self.result = result

    def handle_turn(self, session_id, text, **kw):
        return self.result


class _Mem:
    def append_turn(self, *_a, **_k):
        pass

    def set_authenticated(self, *a, **k):
        pass


def _inbox(result):
    links = []
    h = InboxHandler(_Orch(result), _Mem(), on_filter=lambda f, label: links.append((f, label)))
    h.audit = SimpleNamespace(write=lambda **k: None)
    h.set_selection("s1", "evt-1")
    return h, links


def test_inbox_chat_listing_pushes_the_events_tab_link():
    result = SimpleNamespace(answer="2 pending (unacknowledged) events on bed Unit1-Bed02:\n- …",
                             declined=False, history_filter={"status": "pending",
                                                             "bed": "Unit1-Bed02"})
    h, links = _inbox(result)
    assert h.respond("pending events on bed 2", session_id="s1").startswith("2 pending")
    assert links == [({"status": "pending", "bed": "Unit1-Bed02"},
                      "2 pending (unacknowledged) events on bed Unit1-Bed02")]


def test_other_answers_push_no_link():
    h, links = _inbox(SimpleNamespace(answer="MEWS 6.", declined=False, history_filter=None))
    h.respond("what is the MEWS", session_id="s1")
    assert links == []
