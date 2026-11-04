"""Simulate tab: `cli.simulate` (plan / tally / run) and `POST /simulate`."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from cli.simulate import (
    CALL_CLASSES,
    FP_CLASS,
    MAX_SAMPLES,
    hospital_for,
    plan,
    run,
    tally,
    targets,
)
from ingest.hdf5_reader import read_hdf5_file
from tests.test_real_samples import _make_package


def test_plan_splits_false_positives_from_call_worthy_classes():
    pick = plan(10, 20, random.Random(1))
    assert sum(pick.values()) == 10 and pick[FP_CLASS] == 2
    assert set(pick) - {FP_CLASS} <= set(CALL_CLASSES)
    assert plan(4, 0, random.Random(1)).get(FP_CLASS) is None
    assert plan(3, 100, random.Random(1)) == {FP_CLASS: 3}
    for bad in ((0, 10), (MAX_SAMPLES + 1, 10), (5, -1), (5, 101)):
        with pytest.raises(ValueError):
            plan(*bad, random.Random(1))


def test_tally_separates_model_errors_from_the_vitals_override():
    VT = "VENTRICULAR_TACHYCARDIA"
    lines = [
        {"ground_truth": VT, "call": True, "gate": "ok"},
        {"ground_truth": VT, "call": False, "gate": "low_confidence_arrhythmia"},
        {"ground_truth": VT, "call": False, "gate": "vitals_alert"},
        {"ground_truth": FP_CLASS, "call": False, "gate": "suppressed_false_positive"},
        {"ground_truth": FP_CLASS, "call": True, "gate": "ok"},           # model called it VT
        {"ground_truth": FP_CLASS, "call": True, "gate": "fp_override"},  # vitals rang it
        {"ground_truth": FP_CLASS, "call": True, "gate": "vitals_alert"},  # unsure NSR, vitals rang
    ]
    assert tally(lines) == {"expected_calls": 3, "tp_calls": 1, "missed_worklist": 1, "missed": 1,
                            "fp_suppressed": 1,
                            "fp_called_model": 1, "fp_called_vitals": 2}


def test_run_stamps_the_run_id_and_scores_the_ingest(tmp_path, monkeypatch):
    import cli.simulate

    pkg_dir = _make_package(tmp_path / "pkg")  # test split: VT x2, AF x1 (+ a missing PVC)
    monkeypatch.setattr(cli.simulate, "CALL_CLASSES", ("VENTRICULAR_TACHYCARDIA",))
    out = tmp_path / "out"
    seen = {}

    def fake_ingest(out_dir, redis_url):
        windows = [w for f in sorted(Path(out_dir).glob("*.h5")) for w in read_hdf5_file(f)]
        seen["runs"] = {w.provenance.run for w in windows}
        return [{"published": "1-0", "patient": w.patient_ref, "event_type": "X",
                 "ground_truth": w.ground_truth.condition, "call": True, "gate": "ok"}
                for w in windows]

    result = run(2, 0, seed=3, package=str(pkg_dir), out_dir=str(out), checkpoints=("x.pt",),
                 run_number=lambda: 7, ingest=fake_ingest)
    assert result["run"].startswith("r7-") and result["run"].endswith("-n2-fp0")
    assert seen["runs"] == {result["run"]}  # every written event carries this run's id
    s = result["summary"]
    assert s["expected_calls"] == len(result["events"]) == s["tp_calls"]
    assert sum(result["pick"].values()) == 2


def test_two_hospitals_split_by_patient_and_ingest_to_their_own_bus(tmp_path, monkeypatch):
    import cli.simulate

    urls = {"h1": "redis://localhost:6379/0", "h2": "redis://localhost:6380/0"}
    monkeypatch.setenv("SIMULATE_TARGETS", ",".join(f"{h}={u}" for h, u in urls.items()))
    monkeypatch.setattr(cli.simulate, "CALL_CLASSES", ("VENTRICULAR_TACHYCARDIA",))
    calls = []

    def fake_ingest(out_dir, redis_url):
        calls.append((Path(out_dir).name, redis_url))
        return [{"event_id": w.event_id, "patient": w.patient_ref,
                 "ground_truth": w.ground_truth.condition, "call": True, "gate": "ok"}
                for f in Path(out_dir).glob("*.h5") for w in read_hdf5_file(f)]

    result = run(2, 0, seed=3, package=str(_make_package(tmp_path / "pkg")),
                 out_dir=str(tmp_path / "out"), checkpoints=("x.pt",), run_number=lambda: 1,
                 ingest=fake_ingest)
    assert result["events"] and all(urls[h] == u for h, u in calls)  # each share -> its own bus
    for e in result["events"]:  # a patient always lands in the same hospital
        assert e["hospital"] == hospital_for(e["patient"], ["h1", "h2"])
    assert hospital_for("PT1", ["h2", "h1"]) == hospital_for("PT1", ["h1", "h2"])


def test_targets_default_to_this_hospital_and_reject_bad_entries(monkeypatch):
    monkeypatch.delenv("SIMULATE_TARGETS", raising=False)
    assert len(targets()) == 1
    monkeypatch.setenv("SIMULATE_TARGETS", "h1=http://x")
    with pytest.raises(ValueError):
        targets()


def test_run_refuses_the_stub_model(tmp_path):
    with pytest.raises(ValueError, match="stub"):
        run(1, 0, package=str(_make_package(tmp_path / "pkg")), checkpoints=())


# --- POST /simulate ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.audit import AuditLog  # noqa: E402
from live.gateway import create_app  # noqa: E402
from tests.test_gateway_perf import _CFG, _token  # noqa: E402


def _client(tmp_path, simulator):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    return TestClient(create_app(_CFG, audit=audit, simulator=simulator)), audit


def test_simulate_needs_a_session_and_reports_bad_input(tmp_path):
    calls = Counter()

    def sim(samples, fp_percent):
        calls["n"] += 1
        plan(samples, fp_percent, random.Random(0))  # the real bounds check
        return {"run": "r1-0000-n1-fp0", "events": [], "summary": {}}

    client, audit = _client(tmp_path, sim)
    assert client.post("/simulate", json={"session": "bad", "samples": 1}).status_code == 401
    assert calls["n"] == 0 and audit.read_all()[-1]["outcome"] == "unauthorized"
    assert client.post("/simulate", json={"session": _token(), "samples": 99}).status_code == 400
    ok = client.post("/simulate", json={"session": _token(), "samples": 1, "fp_percent": 0})
    assert ok.status_code == 200 and ok.json()["run"] == "r1-0000-n1-fp0"
    assert audit.read_all()[-1]["action"] == "simulate"
