"""Offline tests for operational-result rendering — readable text, not raw dicts."""

from __future__ import annotations

from orchestrator.orchestrator import _answer_operational, _render_operational


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


# --- deterministic operational answer (bypasses the LLM) ---


def test_answer_operational_single_row_prose():
    out = _answer_operational([{"patient": "PT8620", "bed": "Unit1-Bed01", "unit": "Unit1",
                                "event_type": "AV_BLOCK_2_TYPE2", "criticality": "High",
                                "mews_risk": "Low", "hr": 68.0, "sbp": 118.0, "dbp": 85.0,
                                "spo2": 97.0, "rr": 18.0, "temp": 98.6, "ts": 1782627240.0,
                                "note": None}])
    assert out.startswith("At 2026-")                         # leads with the time
    assert "patient PT8620 on bed Unit1-Bed01 in unit Unit1" in out
    assert "had an event type AV_BLOCK_2_TYPE2; criticality High; MEWS risk Low" in out
    assert "The vitals at this event were hr 68; sbp 118; dbp 85; S P O 2 97" in out   # spo2 spelled
    assert "{" not in out and "note" not in out               # no objects / null fields


def test_answer_operational_multi_row_one_sentence_per_line():
    out = _answer_operational([
        {"patient": "PT1", "bed": "Unit1-Bed01", "event": "VENTRICULAR_FIBRILLATION",
         "criticality": "Critical", "ts": 1782627240.0},
        {"patient": "PT2", "bed": "Unit1-Bed02", "event": "SVT", "criticality": "High",
         "ts": 1782627240.0},
    ])
    lines = out.splitlines()
    assert lines[0] == "2 matching records."
    assert "patient PT1 on bed Unit1-Bed01 had an event type VENTRICULAR_FIBRILLATION" in lines[1]
    assert "patient PT2 on bed Unit1-Bed02 had an event type SVT" in lines[2]
    assert all(line.endswith(".") for line in lines[1:])      # each its own sentence -> TTS pause


def test_answer_operational_empty():
    assert _answer_operational([]) == "No matching records."
    assert _answer_operational(None) == "No matching records."
