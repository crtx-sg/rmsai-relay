"""Criticality lookup (G1)."""

from __future__ import annotations

from common.criticality import assess_criticality, at_least, criticality


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
