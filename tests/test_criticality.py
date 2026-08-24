"""Criticality lookup (G1) + what an alert is allowed to claim (alert_basis)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from common.config import DEFAULT
from common.criticality import alert_basis, assess_criticality, at_least, criticality


def test_vf_is_critical_regardless_of_mews():
    assert criticality("VENTRICULAR_FIBRILLATION", "Low") == "Critical"


# --- configurable escalation (assess_criticality) ----------------------------------------------


def test_any_non_normal_event_escalates_to_high():
    # Intrinsically Low events (RBBB, PAC, AV_BLOCK_1) become High purely by not being NORMAL_SINUS.
    for ev in ("RBBB", "PAC", "AV_BLOCK_1", "SINUS_BRADYCARDIA"):
        assert assess_criticality(ev, "Low", 0, False) == "High", ev


def test_normal_sinus_stays_low_when_stable_and_low_mews():
    assert assess_criticality("NORMAL_SINUS", "Low", 0, False) == "Low"


def test_high_mews_score_escalates_normal_sinus():
    # Below threshold -> Low; at/above threshold -> High.
    assert assess_criticality("NORMAL_SINUS", "Low", 2, False, mews_threshold=3) == "Low"
    assert assess_criticality("NORMAL_SINUS", "Low", 3, False, mews_threshold=3) == "High"


def test_deteriorating_vitals_escalate_normal_sinus():
    assert assess_criticality("NORMAL_SINUS", "Low", 0, True) == "High"
    # ...unless trend-based escalation is disabled.
    assert assess_criticality("NORMAL_SINUS", "Low", 0, True,
                              escalate_on_deteriorating=False) == "Low"


def test_escalation_never_lowers_a_critical_event():
    # VT is intrinsically Critical; escalation only raises to High, so it must stay Critical.
    assert assess_criticality("VENTRICULAR_TACHYCARDIA", "Low", 0, False) == "Critical"


def test_configurable_normal_baseline():
    # Treat AV_BLOCK_1 as the "normal" baseline -> it no longer escalates; NORMAL_SINUS now does.
    assert assess_criticality("AV_BLOCK_1", "Low", 0, False, normal_event="AV_BLOCK_1") == "Low"
    assert assess_criticality("NORMAL_SINUS", "Low", 0, False, normal_event="AV_BLOCK_1") == "High"


def test_mews_can_raise_a_low_event():
    # PAC is intrinsically Low, but a Critical MEWS dominates.
    assert criticality("PAC", "Critical") == "Critical"


def test_more_severe_wins():
    assert criticality("ATRIAL_FIBRILLATION", "Low") == "High"  # event dominates
    assert criticality("RBBB", "High") == "High"  # mews dominates


def test_normal_sinus_low():
    assert criticality("NORMAL_SINUS", "Low") == "Low"


def test_at_least():
    assert at_least("Critical", "High")
    assert at_least("High", "High")
    assert not at_least("Medium", "High")


# --- alert basis: what an alert is allowed to claim ---------------------------------------------


class _Ev:
    """Duck-typed DeviceEvent: only the fields `alert_basis` reads."""

    def __init__(self, event_type, confidence, *, fp=False, mews=0, deteriorating=False):
        self.event_type = event_type
        self.confidence = confidence
        self.is_false_positive = fp
        trend = SimpleNamespace(direction="deteriorating" if deteriorating else "stable")
        self.analysis = SimpleNamespace(mews=SimpleNamespace(score=mews, risk="Low"),
                                        vital_trends={"HR": trend})


def test_confident_rhythm_is_its_own_basis():
    # Bad vitals do not demote a rhythm the classifier stands behind.
    assert alert_basis(_Ev("ATRIAL_FIBRILLATION", 0.92, deteriorating=True), DEFAULT) == ("rhythm", "")
    assert alert_basis(_Ev("ATRIAL_FIBRILLATION", 0.92, mews=4), DEFAULT)[0] == "rhythm"


def test_uncertain_rhythm_with_bad_vitals_is_vitals_driven():
    # The alert still goes out, but it rests on the vital — which the caller must name instead of
    # asserting a rhythm at 30% confidence.
    basis, why = alert_basis(_Ev("ATRIAL_FIBRILLATION", 0.30, deteriorating=True), DEFAULT)
    assert basis == "vitals" and "deteriorating" in why
    basis, why = alert_basis(_Ev("ATRIAL_FIBRILLATION", 0.30, mews=4), DEFAULT)
    assert basis == "vitals" and "MEWS 4" in why


def test_uncertain_rhythm_with_calm_vitals_has_no_basis_at_all():
    # Nothing to re-base onto; should_call withholds this one.
    assert alert_basis(_Ev("ATRIAL_FIBRILLATION", 0.30), DEFAULT) == ("rhythm", "")


def test_false_positive_with_bad_vitals_is_vitals_driven():
    assert alert_basis(_Ev("NORMAL_SINUS", 0.99, fp=True, mews=4), DEFAULT)[0] == "vitals"
    # ...unless the FP override is switched off, which returns it to a rhythm (and a no-call).
    off = replace(DEFAULT, criticality_fp_override_on_vitals=False)
    assert alert_basis(_Ev("NORMAL_SINUS", 0.99, fp=True, mews=4), off) == ("rhythm", "")


def test_uncertain_normal_sinus_with_bad_vitals_is_also_vitals_driven():
    # Below fp_suppress_min_confidence a NORMAL_SINUS is never suppressed, so it isn't flagged a
    # false positive — but a normal rhythm is not a finding either. Any alert on one rests on the
    # vitals, or the row would show a bare NORMAL_SINUS at High criticality with no stated reason.
    basis, why = alert_basis(_Ev("NORMAL_SINUS", 0.65, fp=False, mews=4), DEFAULT)
    assert basis == "vitals" and "MEWS 4" in why


def test_normal_sinus_with_calm_vitals_is_not_an_alert_at_all():
    assert alert_basis(_Ev("NORMAL_SINUS", 0.65, fp=False), DEFAULT) == ("rhythm", "")
