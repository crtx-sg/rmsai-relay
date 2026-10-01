"""E5 client: the companion app's dashboard/detail/worklist rendering, run in Node against a stub DOM.

The payloads are built by the real server code (`build_perf_view`, the `/event-info` shape) from
synthetic rows, so a renamed key on either side fails here, which is how the "wrong class undefined"
bug was caught. Skipped when `node` isn't installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from common.ecg_model_stub import StubECGModel
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.explain import explain_event
from orchestrator.perf_view import build_perf_view

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = next((_ROOT / "data" / "fixtures").glob("*.h5"))
VT, NSR, AF, LBBB = "VENTRICULAR_TACHYCARDIA", "NORMAL_SINUS", "ATRIAL_FIBRILLATION", "LBBB"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _row(eid, pred, truth, outcome, gate=True, dataset="mitbih"):
    return {"patient": "PT935761", "id": eid, "timestamp": 1.0, "processed_at": 1_759_300_000.0,
            "event_type": pred, "confidence": 0.58, "ground_truth_condition": truth,
            "criticality": "High", "model_id": "ecg_transconv:v2", "model_classes": None,
            "eval_outcome": outcome, "alert_gate": gate, "alert_reason_code": "vitals_alert",
            "why": "Shown because the vitals warrant it, not the rhythm.", "delivered_app": False,
            "source_kind": "ecg_sigma", "source_dataset": dataset, "source_record": f"{dataset}:106",
            "source_sample": 10038, "source_split": "test"}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render")
    rows = [_row("e1", VT, VT, "TP"), _row("e2", NSR, NSR, "TN", gate=False),
            _row("e3", AF, LBBB, "TP_WRONG_CLASS", dataset="incart")]
    ev = process_window(next(read_hdf5_file(_FIXTURE)), StubECGModel(), MewsVitalsAnalysis())
    x = explain_event(ev)
    info = {"event_id": "e2", "patient": "PT935761", "predicted": NSR, "confidence": 0.58,
            "truth": NSR, "outcome": "TN", "unscorable_reason": None, "model_id": "stub",
            "criticality": x["criticality"]["level"], "why": x["headline"], "explanation": x,
            "source": "MIT-BIH 106 @10038 (test split)",
            "provenance": {"kind": "ecg_sigma", "dataset": "mitbih", "record": "mitbih:106",
                           "split": "test"},
            "delivery": {"app": False}, "processed_at": 1.0, "timestamp": 1.0}
    row_msg = {"type": "event", "event_id": "e9", "patient": "PT992591", "event_type": VT,
               "ts": 1.0, "criticality": "Critical", "status": "reported", "links": {},
               "confidence": 0.5, "why": "Shown because the vitals warrant it.",
               "source": "INCART I05 @263874 (test split)", "truth": VT, "outcome": "TP"}
    paths = []
    for name, data in (("metrics", {**build_perf_view(rows), "filters": {}}), ("info", info),
                       ("row", row_msg)):
        p = tmp / f"{name}.json"
        p.write_text(json.dumps(data))
        paths.append(str(p))
    out = subprocess.run(["node", str(_ROOT / "tests/js/render_smoke.js"),
                          str(_ROOT / "app/app.js"), *paths],
                         capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def test_no_template_leaks(rendered):
    assert not rendered["leaks"]  # no "${" or "undefined" anywhere in the rendered HTML


def test_performance_view(rendered):
    m = rendered["models"]
    assert "ecg_transconv:v2" in m and "Small sample (3 scored)" in m
    assert "TP 2 (wrong class 1) · FP 0 · FN 0 · TN 1" in m
    assert 'class="diag"' in m and 'class="off"' in m and 'table class="pc"' in m
    assert rendered["rows"].count("<tr ") == 3 and "oc-TP_WRONG_CLASS" in rendered["rows"]
    assert sorted(rendered["datasetOptions"]) == ["incart", "mitbih"]


def test_detail_panel(rendered):
    i = rendered["info"]
    assert "Rhythm" in i and "Vitals" in i and "Criticality &amp; decision" in i and "Data source" in i
    assert "MIT-BIH 106 @10038 (test split)" in i and "oc-TN" in i and "stub" in i


def test_worklist_row_shows_why_source_and_outcome(rendered):
    r = rendered["worklistRow"]
    assert 'class="src"' in r and "INCART I05" in r
    assert "oc-TP" in r and "TP ✓" in r
    assert 'class="why"' in r and "Shown because the vitals warrant it." in r
