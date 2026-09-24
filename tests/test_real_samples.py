"""Real-ECG sample curation from an ecgpkg package → reader-ready HDF5 (cli.real_samples)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import h5py
import pytest

from cli.ingest import main as ingest_main
from cli.real_samples import main as samples_main
from ingest.hdf5_reader import read_hdf5_file
from ingest.real_samples import (
    SampleError,
    load_package,
    parse_pick,
    select_events,
    write_samples,
)

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_COLS = ["event_uid", "split", "label", "dataset", "subject_id", "h5_relpath", "event_key"]


def _make_package(root: Path) -> Path:
    """Two records (copies of the fixture, events 1001/1002 each) with a hand-labelled manifest."""
    for name in ("rec_a", "rec_b"):
        (root / "data" / "ds").mkdir(parents=True, exist_ok=True)
        shutil.copy(_FIXTURE, root / "data" / "ds" / f"{name}.h5")
    rows = [
        # uid, split, label, dataset, subject, relpath, key
        ("a1", "test", "VENTRICULAR_TACHYCARDIA", "ds", "ds:a", "data/ds/rec_a.h5", "event_1001"),
        ("a2", "train", "VENTRICULAR_TACHYCARDIA", "ds", "ds:a", "data/ds/rec_a.h5", "event_1002"),
        ("b1", "test", "ATRIAL_FIBRILLATION", "ds", "ds:b", "data/ds/rec_b.h5", "event_1001"),
        ("b2", "test", "VENTRICULAR_TACHYCARDIA", "ds", "ds:b", "data/ds/rec_b.h5", "event_1002"),
        ("x1", "excluded", "", "ds", "ds:b", "data/ds/rec_b.h5", "event_1002"),
        ("m1", "test", "PVC", "ds", "ds:c", "data/ds/missing.h5", "event_1001"),
    ]
    with (root / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLS)
        w.writerows(rows)
    (root / "package.json").write_text(json.dumps({
        "format": "ecgpkg", "package_version": "vT", "manifest_sha256": "abc123def456" + "0" * 52,
        "classes": ["ATRIAL_FIBRILLATION", "PVC", "VENTRICULAR_TACHYCARDIA"],
    }))
    return root


@pytest.fixture
def pkg(tmp_path):
    return load_package(_make_package(tmp_path / "pkg"))


def test_parse_pick():
    assert parse_pick("ventricular_tachycardia:2, PVC") == {"VENTRICULAR_TACHYCARDIA": 2, "PVC": 1}
    for bad in ("VT:2", "PVC:0", "PVC:x", ""):
        with pytest.raises(SampleError):
            parse_pick(bad)


def test_selection_is_test_split_only_and_deterministic(pkg):
    sel = select_events(pkg, {"VENTRICULAR_TACHYCARDIA": 5}, seed=7)
    assert {r["event_uid"] for r in sel.rows} == {"a1", "b2"}  # a2 is train → never drawn
    assert sel.shortfall == {"VENTRICULAR_TACHYCARDIA": (5, 2)}
    one = [r["event_uid"] for r in select_events(pkg, {"VENTRICULAR_TACHYCARDIA": 1}, seed=7).rows]
    assert one == [r["event_uid"] for r in select_events(pkg, {"VENTRICULAR_TACHYCARDIA": 1},
                                                         seed=7).rows]


def test_written_files_carry_manifest_label_as_ground_truth(pkg, tmp_path):
    sel = select_events(pkg, {"VENTRICULAR_TACHYCARDIA": 2, "ATRIAL_FIBRILLATION": 1, "PVC": 1})
    files = write_samples(pkg, sel, tmp_path / "out")  # PVC's source is missing → skipped, no raise
    assert sorted(f.name for f in files) == ["rec_a.h5", "rec_b.h5"]

    # both records are copies of one fixture, so event ids repeat across files — keep a list
    truth = [w.ground_truth.condition for f in files for w in read_hdf5_file(f)]
    assert sorted(truth) ==["ATRIAL_FIBRILLATION", "VENTRICULAR_TACHYCARDIA",
                                      "VENTRICULAR_TACHYCARDIA"]
    with h5py.File(tmp_path / "out" / "rec_a.h5") as hf:
        assert sorted(k for k in hf if k.startswith("event_")) == ["event_1001"]  # train one left out
        assert hf["metadata"].attrs["ecgpkg_split"] == "test"
        assert hf["metadata"].attrs["ecgpkg_version"] == "vT"
        assert hf["event_1001"].attrs["source_condition"] == "AFIB"  # the fixture's own code, kept


def test_refuses_to_write_inside_package(pkg):
    sel = select_events(pkg, {"ATRIAL_FIBRILLATION": 1})
    with pytest.raises(SampleError):
        write_samples(pkg, sel, pkg.root / "out")


def test_cli_pick_then_ingest_dir_scores(tmp_path, capsys):
    root = _make_package(tmp_path / "pkg")
    out = tmp_path / "real"
    rc = samples_main(["--package", str(root), "pick", "--pick", "VENTRICULAR_TACHYCARDIA:2",
                       "--out", str(out), "--checkpoint"])  # no checkpoint → manifest not checked
    assert rc == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["events"] == 2

    assert ingest_main(["--dir", str(out), "--emit", "stdout", "--checkpoint"]) == 0
    captured = capsys.readouterr()
    lines = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 2 and {ln["ground_truth"] for ln in lines} == {"VENTRICULAR_TACHYCARDIA"}
    summary = next(json.loads(ln) for ln in captured.err.splitlines()
                   if ln.startswith('{"summary"'))["summary"]
    assert summary["events"] == summary["scored"] == 2
    assert summary["model"] == "stub" and "STUB" in captured.err  # never a silent stub score


def test_cli_refuses_manifest_mismatch(tmp_path, monkeypatch, capsys):
    import cli.real_samples as mod

    root = _make_package(tmp_path / "pkg")
    monkeypatch.setattr(mod, "checkpoint_manifest", lambda _c: "ffffffffffff")
    args = ["--package", str(root), "pick", "--pick", "ATRIAL_FIBRILLATION:1",
            "--out", str(tmp_path / "o"), "--checkpoint", "x.pt"]
    assert samples_main(args) == 2
    assert "not held out" in capsys.readouterr().err
    assert samples_main([*args, "--allow-mismatch"]) == 0


def test_cli_list(tmp_path, capsys):
    root = _make_package(tmp_path / "pkg")
    assert samples_main(["--package", str(root), "list"]) == 0
    out = capsys.readouterr().out
    assert "VENTRICULAR_TACHYCARDIA" in out and "ds:2" in out
