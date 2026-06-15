"""Synthetic cohort generator (patients -> bed + history)."""

from __future__ import annotations

from cli.gen_synthetic import build_cohort


def test_cohort_assigns_bed_and_history():
    cohort = build_cohort(["PT1000", "PT1001", "PT1002"])
    assert len(cohort) == 3
    beds = {c["bed"] for c in cohort}
    assert len(beds) == 3  # unique beds
    for c in cohort:
        assert c["history"]["patient_id"] == c["patient_id"]
        assert c["unit"].startswith("Unit")


def test_cohort_is_stable():
    a = build_cohort(["PT1000"])
    b = build_cohort(["PT1000"])
    assert a[0]["history"] == b[0]["history"]
