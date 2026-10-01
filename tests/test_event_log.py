"""E4: the `[event]` log line, the rolling performance tracker, and `cli.model_perf`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from cli.model_perf import group_by_model, main as perf_main, parse_since
from common.ecg_model_stub import StubECGModel
from common.schemas import Provenance
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.event_log import PerfTracker, event_log_line, event_trace, source_label

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
VT, NSR, AF = "VENTRICULAR_TACHYCARDIA", "NORMAL_SINUS", "ATRIAL_FIBRILLATION"


@pytest.mark.parametrize("prov,expect", [
    (Provenance(kind="ecg_sigma", dataset="incart", record="incart:I05", source_sample=263874,
                split="test"), "INCART I05 @263874 (test split)"),
    ({"kind": "ecg_sigma", "dataset": "mitbih", "record": "mitbih:207"}, "MIT-BIH 207"),
    ({"kind": "ecg_sigma", "dataset": "newdb", "record": "x1"}, "NEWDB x1"),
    (Provenance(kind="simulator", device="RMSAI-SimDevice-v2.0"), "simulator"),
    (Provenance(kind="device", device="MX800"), "device MX800"),
    (None, "source unknown"),
])
def test_source_label(prov, expect):
    assert source_label(prov) == expect


def _trace(**kw):
    t = {"patient": "PT992591", "source": "INCART I05 @263874 (test split)", "predicted": VT,
         "confidence": 0.5, "truth": VT, "outcome": "TP", "criticality": "Critical", "gate": True,
         "reason_code": "vitals_alert", "why": "Shown because the vitals warrant it.",
         "model_id": "m1", "model_classes": None}
    t.update(kw)
    return t


def test_event_line_alerted_with_delivery():
    line = event_log_line(_trace(), app=False, call="no_answer", sms="sms_delivered")
    assert line == ("[event] PT992591 · INCART I05 @263874 (test split) · pred "
                    "VENTRICULAR_TACHYCARDIA 50% · truth VENTRICULAR_TACHYCARDIA → TP · Critical · "
                    "alert ✓ vitals_alert · app ✗ · call no_answer · sms delivered · "
                    "why: Shown because the vitals warrant it.")


def test_event_line_unlabelled_and_not_alerted():
    line = event_log_line(_trace(truth=None, gate=False, reason_code="false_positive"), app=True)
    assert "truth —" in line and "alert ✗ false_positive" in line
    assert "app" not in line.split("why:")[0]  # no delivery parts when no alert was due


def test_event_trace_from_a_real_event():
    ev = process_window(next(read_hdf5_file(_FIXTURE)), StubECGModel(), MewsVitalsAnalysis())
    t = event_trace(ev)
    assert t["patient"] == ev.window.patient_ref and t["source"] == "simulator"
    assert t["model_id"] == "stub" and t["truth"] == ev.window.ground_truth.condition
    assert t["outcome"] in {"TP", "TP_WRONG_CLASS", "FP", "FN", "TN", "UNSCORABLE"}
    assert t["why"].startswith(("Shown because", "Not alerted"))


def test_tracker_prints_every_n_labelled_per_model():
    out = []
    tr = PerfTracker(every=2, emit=out.append)
    tr.add(_trace())
    tr.add(_trace(truth=None))          # unlabelled: ignored
    assert out == []
    tr.add(_trace(predicted=NSR, truth=NSR, outcome="TN", gate=False))
    assert len(out) == 1 and "after 2 labelled events · model m1" in out[0]
    assert "TP 1 (wrong class 0) · FP 0 · FN 0 · TN 1" in out[0]
    tr.add(_trace(model_id="m2"))       # a second model gets its own block
    tr.flush(header="on exit")
    assert [o.split("\n")[0] for o in out[1:]] == ["[perf] on exit · model m1",
                                                   "[perf] on exit · model m2"]


def test_tracker_every_zero_only_on_flush():
    out = []
    tr = PerfTracker(every=0, emit=out.append)
    for _ in range(5):
        tr.add(_trace())
    assert out == []
    tr.flush()
    assert len(out) == 1


# --- cli.model_perf --------------------------------------------------------------------------

NOW = datetime(2026, 10, 1, 12, 0).timestamp()


@pytest.mark.parametrize("text,expect", [
    ("30m", NOW - 1800), ("24h", NOW - 86400), ("7d", NOW - 7 * 86400),
    ("1700000000", 1700000000.0), ("2026-10-01", datetime(2026, 10, 1).timestamp()), (None, None),
])
def test_parse_since(text, expect):
    assert parse_since(text, now=NOW) == expect


def test_parse_since_rejects_nonsense():
    with pytest.raises(ValueError, match="--since"):
        parse_since("yesterday-ish", now=NOW)


def _row(model, pred, truth, **kw):
    r = {"patient": "PT1", "id": f"e-{pred}-{truth}", "timestamp": NOW - 99, "processed_at": NOW,
         "event_type": pred, "confidence": 0.8, "ground_truth_condition": truth,
         "model_id": model, "model_classes": None, "alert_gate": True, "eval_outcome": "TP",
         "why": "Shown because…", "source_kind": "ecg_sigma", "source_dataset": "mitbih",
         "source_record": "mitbih:207", "source_sample": 1, "source_split": "test"}
    r.update(kw)
    return r


class FakeDriver:
    def __init__(self, rows):
        self.rows, self.params = rows, None

    def run_read(self, query, **params):
        self.params = params
        return self.rows


def test_group_by_model_keeps_models_apart():
    g = group_by_model([_row("a", VT, VT), _row("b", VT, VT), _row("a", NSR, NSR)])
    assert list(g) == ["a", "b"] and len(g["a"]["rows"]) == 2


def test_model_perf_report(capsys):
    d = FakeDriver([_row("ecg_transconv:v2", VT, VT), _row("ecg_transconv:v2", AF, NSR),
                    _row("stub", NSR, VT)])
    assert perf_main(["--events", "--dataset", "mitbih"], driver=d) == 0
    out = capsys.readouterr().out
    assert "3 labelled event(s) · filters: dataset=mitbih" in out
    assert "== model ecg_transconv:v2" in out and "== model stub" in out
    assert "MIT-BIH 207 @1 (test split)" in out and "why: Shown because…" in out
    assert d.params["dataset"] == "mitbih"


def test_model_perf_json_and_since(capsys):
    d = FakeDriver([_row("m", VT, VT), _row("m", NSR, NSR)])
    assert perf_main(["--json", "--since", "24h"], driver=d) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["m"]["counts"]["TP"] == 1 and data["m"]["counts"]["TN"] == 1
    assert d.params["since"] is not None


def test_model_perf_nothing_labelled(capsys):
    assert perf_main([], driver=FakeDriver([])) == 1
    assert "no labelled events" in capsys.readouterr().err
