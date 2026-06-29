"""Bus payload round-trip: event_to_dict -> dict_to_event preserves what the relay reads."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from common.interfaces import ECGModel
from inference.pipeline import process_window
from inference.serialize import dict_to_event, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.outbound_flow import should_call
from orchestrator.report import spoken_report

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


def _afib_event():
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0
    return process_window(w, _Afib(), MewsVitalsAnalysis())


def test_roundtrip_preserves_relay_fields():
    ev = _afib_event()
    back = dict_to_event(event_to_dict(ev))

    assert back.event_type == ev.event_type
    assert back.confidence == ev.confidence
    assert back.is_false_positive == ev.is_false_positive
    assert back.window.patient_ref == ev.window.patient_ref
    assert back.window.event_id == ev.window.event_id
    assert back.window.event_timestamp == ev.window.event_timestamp
    assert back.window.vitals["HR"].value == 145.0
    assert back.analysis.mews.score == ev.analysis.mews.score
    assert back.analysis.mews.risk == ev.analysis.mews.risk
    assert back.analysis.care_guidance == ev.analysis.care_guidance
    assert back.report_md == ev.report_md


def test_roundtrip_preserves_ecg_plot_ref_and_hr_history():
    from common.schemas import VitalSample  # noqa: PLC0415

    ev = _afib_event()
    ev.window.ecg_plot_ref = "data/plots/evt.png"
    ev.window.vitals_history["HR"] = [VitalSample(value=120, timestamp=1),
                                      VitalSample(value=145, timestamp=2)]
    back = dict_to_event(event_to_dict(ev))
    assert back.window.ecg_plot_ref == "data/plots/evt.png"     # carried on the bus (path only)
    assert [s.value for s in back.window.vitals_history["HR"]] == [120, 145]  # small history carried


def test_roundtrip_sample_rates_are_fractions():
    back = dict_to_event(event_to_dict(_afib_event()))
    assert all(isinstance(r, Fraction) for r in back.window.sample_rates.values())
    assert back.window.sample_rates.get("resp") == Fraction(100, 3)


def test_roundtrip_drops_signals_but_keeps_decision_inputs():
    ev = _afib_event()
    back = dict_to_event(event_to_dict(ev))  # signals excluded from the bus by default
    assert back.window.signals == {}
    # The criticality gate and spoken alert must behave identically on the reconstructed event.
    assert should_call(back) == should_call(ev)
    assert spoken_report(back, bed="ICU/1") == spoken_report(ev, bed="ICU/1")
