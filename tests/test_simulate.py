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

    def fake_ingest(out_dir, redis_url, hospital):
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

    def fake_ingest(out_dir, redis_url, hospital):
        calls.append((hospital, redis_url))
        assert Path(out_dir).name == hospital
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
    return TestClient(create_app(_CFG, audit=audit, simulator=simulator,
                                 simulate_history=tmp_path / "runs.jsonl")), audit


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


def test_each_share_is_scored_with_its_own_hospital_policy(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import cli.simulate

    seen = {}

    def fake_run(cmd, **kw):
        seen["env"], seen["cmd"] = kw.get("env"), cmd
        return SimpleNamespace(returncode=0, stdout='{"event_id": "e1"}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli.simulate._ingest("d", "redis://localhost:6380/0", "h2") == [{"event_id": "e1"}]
    assert seen["env"]["HOSPITAL_ID"] == "h2" and "redis://localhost:6380/0" in seen["cmd"]
    cli.simulate._ingest("d", "redis://x", "default")
    assert seen["env"] is None  # single hospital, no id: the app's own config as is



def test_hospital_app_links_parse_and_drop_bad_entries():
    from live.gateway import hospital_apps

    assert hospital_apps("h1=http://localhost:8080, h2=http://localhost:8090,x=ftp://y,bad") == {
        "h1": "http://localhost:8080", "h2": "http://localhost:8090"}
    assert hospital_apps("") == {}


def test_history_sums_runs_with_wilson_rates(tmp_path):
    from cli.simulate import history, record

    path = tmp_path / "runs.jsonl"
    assert history(path)["runs"] == []  # no file yet
    for run, s in (("r1-0900-n4-fp50", {"expected_calls": 2, "tp_calls": 1, "fp_suppressed": 2}),
                   ("r2-0910-n2-fp50", {"expected_calls": 1, "tp_calls": 1, "fp_called_model": 1})):
        record({"run": run, "events": [{"hospital": "h1"}, {"hospital": "h2"}], "summary": s}, path)
    with path.open("a") as fh:
        fh.write("not json\n")  # skipped, not fatal
    h = history(path)
    assert [r["run"] for r in h["runs"]] == ["r2-0910-n2-fp50", "r1-0900-n4-fp50"]  # newest first
    assert h["runs"][0]["hospitals"] == {"h1": 1, "h2": 1}
    assert (h["rates"]["rang"]["k"], h["rates"]["rang"]["n"]) == (2, 3)
    assert (h["rates"]["fp_filtered"]["k"], h["rates"]["fp_filtered"]["n"]) == (2, 3)


def test_only_the_console_hospital_simulates_and_records(tmp_path, monkeypatch):
    sim = lambda n, fp: {"run": "r1-0000-n1-fp0", "events": [{"hospital": "h1"}],  # noqa: E731
                         "summary": {"expected_calls": 1, "tp_calls": 1}}
    hist = tmp_path / "runs.jsonl"

    def client():
        return TestClient(create_app(_CFG, audit=AuditLog(str(tmp_path / "a.jsonl")),
                                     simulator=sim, simulate_history=hist))

    body = {"session": _token(), "samples": 1}
    monkeypatch.setenv("SIMULATE_CONSOLE", "h9")  # another hospital is the console
    other = client()
    assert other.post("/simulate", json=body).status_code == 403
    assert other.post("/simulate/history", json={"session": _token()}).status_code == 403
    assert not hist.exists()
    monkeypatch.setenv("SIMULATE_CONSOLE", _CFG.hospital_id)  # this one is
    console = client()
    assert console.post("/simulate", json=body).status_code == 200
    h = console.post("/simulate/history", json={"session": _token()}).json()
    assert [r["run"] for r in h["runs"]] == ["r1-0000-n1-fp0"]


def test_one_hospital_failing_keeps_the_others_published_run(tmp_path, monkeypatch):
    import cli.simulate

    monkeypatch.setenv("SIMULATE_TARGETS", "h1=redis://localhost:6379/0,h2=redis://localhost:6380/0")
    monkeypatch.setattr(cli.simulate, "CALL_CLASSES", ("VENTRICULAR_TACHYCARDIA",))
    import itertools

    turn = itertools.cycle(["h1", "h2"])  # the two VT records go one to each hospital
    monkeypatch.setattr(cli.simulate, "hospital_for", lambda p, hs: next(turn))

    def ingest(out_dir, redis_url, hospital):
        if hospital == "h2":
            raise RuntimeError("h2 Redis down")
        return [{"event_id": w.event_id, "patient": w.patient_ref, "call": True, "gate": "ok",
                 "ground_truth": w.ground_truth.condition}
                for f in Path(out_dir).glob("*.h5") for w in read_hdf5_file(f)]

    kw = dict(seed=3, package=str(_make_package(tmp_path / "pkg")), out_dir=str(tmp_path / "out"),
              checkpoints=("x.pt",), run_number=lambda: 1)
    result = run(2, 0, ingest=ingest, **kw)
    assert list(result["errors"]) == ["h2"] and "h2 Redis down" in result["errors"]["h2"]
    assert [e["hospital"] for e in result["events"]] == ["h1"]  # h1's call still counted
    with pytest.raises(RuntimeError):  # every hospital failed: nothing published, say so
        run(2, 0, ingest=lambda *a: (_ for _ in ()).throw(RuntimeError("down")), **kw)


def test_other_hospitals_are_audited_in_their_own_log(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import cli.simulate

    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (seen.update(env=kw["env"]),
                        SimpleNamespace(returncode=0, stdout="", stderr=""))[1])
    other = "zz" if cli.simulate.DEFAULT.hospital_id != "zz" else "yy"
    cli.simulate._ingest("d", "redis://x", other)
    assert seen["env"]["AUDIT_LOG_PATH"] == f"data/audit-{other}.jsonl"
