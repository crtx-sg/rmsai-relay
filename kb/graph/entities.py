"""Shared entity normalization.

`slugify` produces the canonical node `id` from an entity name. Both patient-record ingestion
and document extraction MUST go through this so a patient's condition and the same condition named
in a protocol resolve to **one shared `Condition` node** (the linkage that makes hybrid retrieval
work). A small synonym map folds common variants onto a canonical name first.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Fold surface variants onto one canonical name before slugifying.
_CONDITION_SYNONYMS = {
    "afib": "atrial fibrillation",
    "a fib": "atrial fibrillation",
    "atrial fib": "atrial fibrillation",
    "vfib": "ventricular fibrillation",
    "v fib": "ventricular fibrillation",
    "vtach": "ventricular tachycardia",
    "v tach": "ventricular tachycardia",
    "cad": "coronary artery disease",
    "chf": "heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "htn": "hypertension",
    "mi": "myocardial infarction",
}

_TREATMENT_SYNONYMS = {
    "beta blocker": "beta-blocker",
    "ace inhibitor": "ace-inhibitor",
    "ace-i": "ace-inhibitor",
}


def slugify(name: str) -> str:
    """Canonical lowercase slug id, e.g. 'Atrial Fibrillation' -> 'atrial_fibrillation'."""
    return _NON_ALNUM.sub("_", name.strip().lower()).strip("_")


def canonical_condition(name: str) -> str:
    key = name.strip().lower()
    return _CONDITION_SYNONYMS.get(key, key)


def canonical_treatment(name: str) -> str:
    key = name.strip().lower()
    return _TREATMENT_SYNONYMS.get(key, key)


def condition_id(name: str) -> str:
    return slugify(canonical_condition(name))


def treatment_id(name: str) -> str:
    return slugify(canonical_treatment(name))
