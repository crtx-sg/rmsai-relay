"""Live call 2026-10-17: an incomputable MEWS was spoken as "MEWS 0 (Low)" with "Recommended:
Insufficient vitals…"; "Kindly acknowledges" was not an ack; "[phone beeping]" got an answer."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from common.config import DEFAULT
from common.criticality import assess_criticality, vitals_override
from common.interfaces import ECGModel
from common.schemas import MEWS
from inference.pipeline import process_window
from inference.serialize import dict_to_event, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.report import spoken_report
from voice.handlers import is_filler
from voice.outbound import parse_ack

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.96


def _unknown_mews_event():
    w = next(read_hdf5_file(_FIXTURE))
    for name in ("Systolic", "Temp"):  # what the vios devices don't send
        w.vitals.pop(name, None)
    return process_window(w, _Afib(), MewsVitalsAnalysis())


def test_unknown_mews_is_spoken_as_unavailable_not_low():
    ev = _unknown_mews_event()
    assert ev.analysis.mews.score is None and ev.analysis.mews.risk == "Unknown"
    text = spoken_report(ev, bed="Unit1-Bed15")
    assert "MEWS unavailable, no systolic blood pressure or temperature" in text
    assert "MEWS 0" not in text and "(Low)" not in text and "Insufficient vitals" not in text


def test_unknown_mews_has_no_effect_rhythm_alone_decides():
    # user decision 2026-10-17: unknown neither raises nor lowers criticality
    assert assess_criticality("NORMAL_SINUS", "Unknown", None, False) == "Low"
    assert assess_criticality("ATRIAL_FIBRILLATION", "Unknown", None, False) == "High"
    assert assess_criticality("VENTRICULAR_FIBRILLATION", "Unknown", None, False) == "Critical"
    ev = _unknown_mews_event()
    ev.analysis.vital_trends.clear()  # isolate MEWS from the trend trigger
    assert vitals_override(ev, replace(DEFAULT, criticality_mews_threshold=0)) == (False, "")


def test_unknown_mews_survives_the_bus_round_trip():
    ev = _unknown_mews_event()
    back = dict_to_event(event_to_dict(ev)).analysis.mews
    assert back.score is None and back.risk == "Unknown" and back.missing == ev.analysis.mews.missing


def test_mews_label():
    assert MEWS(score=4, risk="Medium").label() == "MEWS 4 (Medium)"
    assert MEWS(missing=["Systolic"]).label() == "MEWS unavailable (missing Systolic)"


def test_third_person_ack_and_sound_tags():
    assert parse_ack("Kindly acknowledges. Thank you") == "yes"
    assert is_filler("[phone beeping]") and is_filler("[outro jingle]")
    assert not is_filler("[phone beeping] what is the heart rate")
