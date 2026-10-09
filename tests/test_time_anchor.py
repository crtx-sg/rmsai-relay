"""Ingest time re-anchoring (demo): each recording's last event lands at `now`, spacing kept."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.ingest import main as ingest_main
from ingest.hdf5_reader import read_hdf5_file
from ingest.time_anchor import rebase_to_now

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
NOW = 2_000_000_000.0


def test_last_event_is_now_and_all_spacing_is_kept():
    source = list(read_hdf5_file(_FIXTURE))
    shifted = rebase_to_now(source, NOW)
    assert max(w.event_timestamp for w in shifted) == NOW
    delta = NOW - max(w.event_timestamp for w in source)
    for a, b in zip(source, shifted):
        assert b.event_timestamp - a.event_timestamp == pytest.approx(delta)
        assert b.start_timestamp - a.start_timestamp == pytest.approx(delta)
        for k, v in a.vitals.items():
            assert b.vitals[k].timestamp - v.timestamp == pytest.approx(delta)
        for k, hist in a.vitals_history.items():  # MEWS trends read these: spacing must not move
            new = b.vitals_history[k]
            assert [s.timestamp - new[0].timestamp for s in new] == pytest.approx(
                [s.timestamp - hist[0].timestamp for s in hist])
        assert b.signals == a.signals and b.patient_ref == a.patient_ref
    assert rebase_to_now([], NOW) == []


def test_ingest_reports_the_source_time_when_re_anchoring(capsys, monkeypatch):
    monkeypatch.setenv("ECG_PLOT_ENABLED", "false")
    assert ingest_main(["--file", str(_FIXTURE), "--checkpoint", "--time-anchor", "now"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    events = [l for l in lines if "source_ts" in l]
    assert events and all(e["source_ts"] < 1_900_000_000 for e in events)  # the HDF5 time, kept


def test_source_anchor_is_the_default():
    from common.config import DEFAULT

    assert DEFAULT.ingest_time_anchor == "source"
