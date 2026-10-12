"""Event history: the one `pending` definition, filter plumbing, and the cli.events views.

Offline: a fake driver records the query parameters and returns hand-written rows, so the live
demo graph is never touched.
"""

from __future__ import annotations

import pytest

from cli import events as cli_events
from kb.graph.events import is_pending, list_events, normalize_bed


class _Driver:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    def run_read(self, cypher, **params):
        self.calls.append((cypher, params))
        return [dict(r) for r in self.rows]


def _row(**kw):
    base = {"id": "e1", "patient": "PT900001", "bed": "Unit1-Bed02", "unit": "Unit1",
            "timestamp": 1_790_000_000.0, "event_type": "VENTRICULAR_TACHYCARDIA",
            "confidence": 0.8, "criticality": "Critical", "status": "reported",
            "alert_gate": True, "alert_reason_code": "ok"}
    return {**base, **kw}


@pytest.mark.parametrize("row, pending", [
    (_row(), True),                                   # alerted, nobody has acked
    (_row(status="notify_failed"), True),             # the call failed: still waiting on someone
    (_row(status="acknowledged"), False),
    (_row(status="resolved"), False),
    (_row(alert_gate=False), False),                  # withheld: never reached the worklist
    (_row(alert_gate=None, status=None), False),
])
def test_pending_means_alerted_and_not_closed(row, pending):
    assert is_pending(row) is pending


@pytest.mark.parametrize("text, label", [
    ("Unit1-Bed02", "Unit1-Bed02"), ("bed 2", "Bed02"), ("Bed2", "Bed02"), ("bed02", "Bed02"),
    ("BED 12", "Bed12"), (None, None),
])
def test_bed_normalization(text, label):
    assert normalize_bed(text) == label


def test_filters_reach_the_query():
    d = _Driver([_row()])
    rows = list_events(d, patient="PT900001", bed="bed 2", event_type="ventricular tachycardia",
                       status="pending", min_criticality="High", limit=5)
    cypher, p = d.calls[0]
    assert (p["patient"], p["bed"], p["etype"], p["status"]) == (
        "PT900001", "Bed02", "VENTRICULAR_TACHYCARDIA", "pending")
    assert p["levels"] == ["High", "Critical"] and p["closed"] == ["acknowledged", "resolved"]
    assert "LIMIT $limit" in cypher and p["limit"] == 5
    assert "MATCH" in cypher and "SET" not in cypher and "MERGE" not in cypher  # read-only
    assert rows[0]["pending"] is True


def test_unknown_status_is_refused():
    with pytest.raises(ValueError):
        list_events(_Driver(), status="unacked")


def test_cli_lists_with_status_and_delivery(capsys):
    d = _Driver([
        _row(delivered_app=True, delivered_call="answered", status="acknowledged"),
        _row(id="e2", alert_gate=False, alert_reason_code="low_confidence_arrhythmia",
             ground_truth_condition="LBBB", eval_outcome="TP_WRONG_CLASS"),
        _row(id="e3", alert_basis="vitals"),
    ])
    assert cli_events.main(["--bed", "Unit1-Bed02"], driver=d) == 0
    out = capsys.readouterr().out
    assert "3 event(s), 1 pending · filters: bed=Unit1-Bed02" in out
    assert "acknowledged · delivered app, call answered" in out
    assert "withheld (low_confidence_arrhythmia) · truth LBBB → TP_WRONG_CLASS" in out
    assert "PENDING · vitals alert" in out
    assert "event e2" in out


def test_cli_empty_result_exits_1(capsys):
    assert cli_events.main(["--status", "pending"], driver=_Driver()) == 1
    assert "0 event(s)" in capsys.readouterr().out


def test_ask_shows_the_query_rows_and_answer(capsys):
    d = _Driver([_row()])
    assert cli_events.main(["--ask", "what are the events on bed two"], driver=d) == 0
    out = capsys.readouterr().out
    assert "normalized" in out and "Unit1-Bed02" in out
    assert "[events] query      event_history" in out
    assert "[events] rows       1" in out and "[events] answer" in out


def test_ask_without_a_matching_query_says_so(capsys):
    assert cli_events.main(["--ask", "what is the capital of France"], driver=_Driver()) == 1
    assert "no graph query matches" in capsys.readouterr().out
