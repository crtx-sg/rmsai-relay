"""Criticality lookup (G1)."""

from __future__ import annotations

from common.criticality import at_least, criticality


def test_vf_is_critical_regardless_of_mews():
    assert criticality("VENTRICULAR_FIBRILLATION", "Low") == "Critical"


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
