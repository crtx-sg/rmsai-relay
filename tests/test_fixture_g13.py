"""G13 — synthetic realism: committed fixture has physiological vitals + usable history length.

Reads the committed fixture directly with h5py (the full SignalWindow reader is Phase 1). This
guards that Mann-Kendall/MEWS will be meaningful downstream (>=5 history samples) and that the
simulator output is in range.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
_PLAUSIBLE = {
    "HR": (20, 260),
    "Pulse": (20, 260),
    "SpO2": (50, 100),
    "Systolic": (50, 260),
    "Diastolic": (20, 160),
    "RespRate": (4, 60),
    "Temp": (90, 110),  # Fahrenheit
}


def _fixture_files() -> list[Path]:
    return sorted(_FIXTURES.glob("*.h5"))


def test_fixture_present():
    assert _fixture_files(), "no committed .h5 fixture found in data/fixtures/"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.name)
def test_vitals_in_range_and_history_usable(path: Path):
    with h5py.File(path, "r") as f:
        events = [k for k in f if k.startswith("event_")]
        assert events
        for ek in events:
            vitals = f[ek]["vitals"]
            for vname, (lo, hi) in _PLAUSIBLE.items():
                if vname not in vitals:
                    continue
                value = float(vitals[vname]["value"][()])
                assert lo <= value <= hi, f"{ek}/{vname}={value} out of [{lo},{hi}]"
                extras_raw = vitals[vname]["extras"][()]
                extras = json.loads(extras_raw) if extras_raw else {}
                history = extras.get("history", [])
                assert len(history) >= 5, f"{ek}/{vname} history too short ({len(history)})"


def test_expected_facts_match_fixture():
    facts_path = _FIXTURES / "expected_facts.json"
    assert facts_path.exists()
    facts = json.loads(facts_path.read_text())
    with h5py.File(_FIXTURES / facts["file"], "r") as f:
        assert f["metadata"]["patient_id"][()].decode() == facts["patient_id"]
        events = sorted(k for k in f if k.startswith("event_"))
        assert events == sorted(facts["events"])
