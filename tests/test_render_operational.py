"""Offline tests for operational-result rendering — readable text, not raw dicts."""

from __future__ import annotations

from orchestrator.orchestrator import _render_operational


def test_renders_readable_pairs_not_raw_dict():
    rows = [{"event_type": "SVT", "hr": 171.0, "spo2": 98.0, "temp": 97.9}]
    out = _render_operational("vitals_at_patient_last_event", rows)
    assert "event_type: SVT" in out
    assert "hr: 171" in out and "171.0" not in out      # float noise trimmed
    assert "spo2: 98" in out
    assert "{" not in out and "'" not in out             # no Python object leaking through


def test_drops_null_fields_and_formats_timestamp():
    rows = [{"hr": 60.0, "note": None, "ts": 1782519130.0}]
    out = _render_operational("vitals_at_patient_last_event", rows)
    assert "note" not in out                              # None dropped
    assert "ts: 2026-" in out and "UTC" in out            # epoch -> readable date-time


def test_empty_rows_is_no_records():
    assert _render_operational("vitals_at_patient_last_event", []).endswith("(no matching records)")
