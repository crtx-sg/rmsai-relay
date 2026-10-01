"""Why an event is shown: `orchestrator.explain.explain_event`, hand-checked per gate path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from common.config import DEFAULT
from common.schemas import MEWS, ClinicalAnalysis, MEWSComponentScore, VitalTrend
from orchestrator.explain import explain_event

_CFG = replace(DEFAULT, outbound_min_arrhythmia_confidence=0.60, criticality_mews_threshold=3,
               fp_suppress_min_confidence=0.80, criticality_fp_override_on_vitals=True,
               criticality_escalate_on_deteriorating=True, outbound_min_criticality="High")


def _event(event_type, conf, *, mews=1, risk="Low", components=(), trends=None):
    analysis = ClinicalAnalysis(
        mews=MEWS(score=mews, risk=risk,
                  components=[MEWSComponentScore(name=n, value=v, score=s) for n, v, s in components]),
        vital_trends=trends or {},
    )
    return SimpleNamespace(event_type=event_type, confidence=conf, analysis=analysis,
                           is_false_positive=event_type == "NORMAL_SINUS" and conf >= 0.80)


def test_rhythm_finding():
    x = explain_event(_event("VENTRICULAR_TACHYCARDIA", 0.78), _CFG)
    assert x["decision"]["dispatch"] and x["decision"]["reason_code"] == "ok"
    assert x["basis"] == "rhythm" and x["rhythm"]["status"] == "asserted"
    assert x["criticality"]["level"] == "Critical" and x["criticality"]["escalated_by"] == []
    assert x["headline"] == "Shown because: ventricular tachycardia (78% ≥ 60%), Critical."


def test_vitals_driven_rhythm_unconfirmed_lists_mews_components():
    comps = [("Heart Rate", 125, 2), ("Respiratory Rate", 24, 2), ("SpO2", 91, 1), ("Temperature", 98.6, 0)]
    x = explain_event(_event("VENTRICULAR_TACHYCARDIA", 0.50, mews=6, risk="High", components=comps),
                      _CFG)
    assert x["decision"]["reason_code"] == "vitals_alert" and x["basis"] == "vitals"
    assert x["rhythm"]["status"] == "unconfirmed"
    assert [c["name"] for c in x["vitals"]["mews"]["components"]] == [
        "Heart Rate", "Respiratory Rate", "SpO2"]  # zero-score vitals dropped, highest first
    assert x["headline"] == (
        "Shown because the vitals warrant it: MEWS 6 ≥ 3 (Heart Rate 2, Respiratory Rate 2, SpO2 1). "
        "Rhythm ventricular tachycardia unconfirmed (50% < 60%). Critical.")


def test_vitals_driven_normal_rhythm_names_each_trend_with_direction():
    trends = {"HR": VitalTrend(direction="deteriorating", p=0.012, slope=1.8),
              "SpO2": VitalTrend(direction="deteriorating", p=0.03, slope=-0.4),
              "Temp": VitalTrend(direction="stable", p=0.6, slope=0.01)}
    x = explain_event(_event("NORMAL_SINUS", 0.91, trends=trends), _CFG)
    assert x["decision"]["reason_code"] == "fp_override" and x["basis"] == "vitals"
    assert x["vitals"]["deteriorating"] == [
        {"vital": "HR", "direction": "rising", "p": 0.012},
        {"vital": "SpO2", "direction": "falling", "p": 0.03}]
    assert x["criticality"] == {"level": "High", "base": "Low",
                                "escalated_by": ["deteriorating vitals"]}
    assert x["headline"] == ("Shown because the vitals warrant it, not the rhythm: HR rising "
                             "(p=0.012), SpO₂ falling (p=0.03). Rhythm reads normal (91%). High.")


def test_every_trigger_listed_not_just_the_first():
    # vitals_override reports only "MEWS ≥ 3"; the explanation also names the deteriorating trend
    trends = {"RespRate": VitalTrend(direction="deteriorating", p=0.02, slope=0.9)}
    x = explain_event(_event("NORMAL_SINUS", 0.91, mews=4, risk="Medium",
                             components=[("Heart Rate", 120, 2), ("SpO2", 92, 2)], trends=trends), _CFG)
    assert x["vitals"]["mews"]["triggered"] and x["vitals"]["trend_triggered"]
    assert "MEWS 4 ≥ 3 (Heart Rate 2, SpO2 2); resp rate rising (p=0.02)" in x["headline"]
    assert x["criticality"]["escalated_by"] == ["MEWS 4 ≥ 3", "deteriorating vitals"]


def test_not_alerted_paths():
    fp = explain_event(_event("NORMAL_SINUS", 0.91), _CFG)
    assert not fp["decision"]["dispatch"] and fp["basis"] == "none"
    assert fp["headline"] == "Not alerted: rhythm reads normal (91%) and the vitals are calm."

    low = explain_event(_event("ATRIAL_FIBRILLATION", 0.45), _CFG)
    assert low["decision"]["reason_code"] == "low_confidence_arrhythmia"
    assert low["headline"] == ("Not alerted: atrial fibrillation only 45% (< 60%) and the vitals "
                               "are calm.")

    below = explain_event(_event("ATRIAL_FIBRILLATION", 0.9),
                          replace(_CFG, outbound_min_criticality="Critical"))
    assert below["decision"]["reason_code"] == "below_threshold"
    assert below["headline"] == ("Not alerted: atrial fibrillation, criticality High is below the "
                                 "alert threshold.")


def test_headline_readability_caps_trends_and_keeps_acronyms():
    trends = {k: VitalTrend(direction="deteriorating", p=p, slope=1.0)
              for k, p in (("HR", 4e-9), ("RespRate", 0.02), ("Temp", 0.04), ("Diastolic", 0.001))}
    x = explain_event(_event("RBBB", 0.32, trends=trends), _CFG)
    h = x["headline"]
    assert "HR rising (p<0.001), diastolic BP rising (p=0.001), resp rate rising (p=0.02) (+1 more)" in h
    assert "Rhythm RBBB unconfirmed" in h
    assert len(x["vitals"]["deteriorating"]) == 4  # structured form keeps every trend


def test_rhythm_finding_mentions_abnormal_vitals_too():
    x = explain_event(_event("ATRIAL_FIBRILLATION", 0.9, mews=5, risk="High",
                             components=[("Heart Rate", 140, 3), ("SpO2", 90, 2)]), _CFG)
    assert x["basis"] == "rhythm"
    assert x["headline"] == ("Shown because: atrial fibrillation (90% ≥ 60%), High; vitals also "
                             "abnormal: MEWS 5 ≥ 3 (Heart Rate 3, SpO2 2).")


# --- the contract fields that feed it: filled from the vendored analysis, carried over the bus ---

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


def test_analysis_carries_mews_components_and_trend_slope():
    from inference.vitals_analysis import MewsVitalsAnalysis
    from ingest.hdf5_reader import read_hdf5_file

    a = MewsVitalsAnalysis().analyze(next(read_hdf5_file(_FIXTURE)), "ATRIAL_FIBRILLATION")
    assert a.mews.components and sum(c.score for c in a.mews.components) == a.mews.score
    assessed = [t for t in a.vital_trends.values() if t.direction != "insufficient_data"]
    assert assessed and all(t.slope is not None for t in assessed)


def test_new_fields_survive_the_bus_and_old_payloads_still_parse():
    from inference.pipeline import process_window
    from inference.serialize import dict_to_event, event_to_dict
    from inference.vitals_analysis import MewsVitalsAnalysis
    from ingest.hdf5_reader import read_hdf5_file
    from common.ecg_model_stub import StubECGModel

    ev = process_window(next(read_hdf5_file(_FIXTURE)), StubECGModel(), MewsVitalsAnalysis())
    payload = event_to_dict(ev)
    back = dict_to_event(payload)
    assert back.analysis.mews.components == ev.analysis.mews.components
    assert {n: t.slope for n, t in back.analysis.vital_trends.items()} == {
        n: t.slope for n, t in ev.analysis.vital_trends.items()}
    # a producer from before the fields existed
    payload["mews"].pop("components")
    for t in payload["vital_trends"].values():
        t.pop("slope")
    old = dict_to_event(payload)
    assert old.analysis.mews.components == [] and all(
        t.slope is None for t in old.analysis.vital_trends.values())
