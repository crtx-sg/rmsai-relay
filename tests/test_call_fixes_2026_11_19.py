"""Live call 2026-11-19 (h1, VF 57 %, MEWS 9): the alert said "Critical criticality" and "Critical:
VF detected — initiate ACLS" next to "treat the rhythm as unconfirmed", and read out ">="."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from common.config import DEFAULT
from common.interfaces import ECGModel
from common.schemas import MEWS
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.report import spoken_report

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _Vf(ECGModel):
    def predict(self, window):
        return "VENTRICULAR_FIBRILLATION", 0.57


def test_an_unconfirmed_vf_alert_says_its_guidance_only_holds_if_confirmed():
    ev = process_window(next(read_hdf5_file(_FIXTURE)), _Vf(), MewsVitalsAnalysis())
    ev.analysis.mews = MEWS(score=9, risk="Critical")
    ev.analysis.care_guidance[:] = ["Critical: VF detected — initiate ACLS protocol"]
    text = spoken_report(ev, bed="Unit2-Bed01", config=replace(DEFAULT, criticality_mews_threshold=3))
    assert "Possible ventricular fibrillation, rated critical," in text
    assert "criticality" not in text and ">=" not in text and "at or above threshold 3" in text
    assert "Recommended" not in text and "Critical:" not in text
    assert text.endswith("treat the rhythm as unconfirmed. If it is confirmed: VF detected — "
                         "initiate ACLS protocol.")


def test_block_and_bundle_rhythms_are_spoken_as_clinicians_say_them():
    from orchestrator.report import spoken_event  # noqa: PLC0415

    assert spoken_event("AV_BLOCK_1") == "first-degree AV block"  # was "av block 1"
    assert spoken_event("AV_BLOCK_2_TYPE2") == "second-degree AV block, Mobitz type 2"
    assert spoken_event("LBBB") == "left bundle branch block"
    assert spoken_event("VENTRICULAR_FIBRILLATION") == "ventricular fibrillation"
    assert spoken_event("PVC") == "PVC" and spoken_event(None) == "an event"
