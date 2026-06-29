"""event_type vocabulary: parity with upstream + helpers."""

from __future__ import annotations

import pytest

from common.event_types import (
    CLASS_NAMES,
    CONDITION_CODE_TO_NAME,
    EventType,
    condition_code_to_name,
    event_type_from_text,
    is_false_positive,
    is_valid_event_type,
    load_upstream_class_names,
    load_upstream_condition_codes,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the last AFib event", "ATRIAL_FIBRILLATION"),
        ("atrial fibrillation", "ATRIAL_FIBRILLATION"),
        ("a-fib", "ATRIAL_FIBRILLATION"),
        ("tachycardia", "SINUS_TACHYCARDIA"),
        ("ventricular tachycardia", "VENTRICULAR_TACHYCARDIA"),
        ("v-tach", "VENTRICULAR_TACHYCARDIA"),
        ("supraventricular tachycardia", "SVT"),   # \b keeps it off "ventricular tachycardia"
        ("SVT", "SVT"),
        ("bradycardia", "SINUS_BRADYCARDIA"),
        ("vfib", "VENTRICULAR_FIBRILLATION"),
        ("ST elevation", "ST_ELEVATION"),
        ("STEMI", "ST_ELEVATION"),
        ("left bundle branch block", "LBBB"),
        ("mobitz 2", "AV_BLOCK_2_TYPE2"),
        ("first degree av block", "AV_BLOCK_1"),
    ],
)
def test_event_type_from_text(text, expected):
    assert event_type_from_text(text) == expected
    assert is_valid_event_type(expected)


def test_event_type_from_text_none_when_absent():
    assert event_type_from_text("what are the outstanding action items") is None
    assert event_type_from_text("") is None


# A natural-language phrase for every one of the 16 classes. Guards parameterization: every event
# type a clinician might name must resolve, so the event-scoped intents work for any of them (not
# just AFib/Tachycardia). If a class is added/renamed without a trigger, this fails.
_PHRASE_PER_CLASS = {
    "NORMAL_SINUS": "normal sinus rhythm", "SINUS_BRADYCARDIA": "bradycardia",
    "SINUS_TACHYCARDIA": "sinus tachycardia", "ATRIAL_FIBRILLATION": "afib",
    "ATRIAL_FLUTTER": "atrial flutter", "PAC": "PAC", "SVT": "SVT", "PVC": "PVC",
    "VENTRICULAR_TACHYCARDIA": "v-tach", "VENTRICULAR_FIBRILLATION": "vfib",
    "LBBB": "left bundle branch block", "RBBB": "right bundle branch block",
    "AV_BLOCK_1": "first degree AV block", "AV_BLOCK_2_TYPE1": "mobitz 1",
    "AV_BLOCK_2_TYPE2": "mobitz 2", "ST_ELEVATION": "ST elevation",
}


def test_every_class_has_a_resolving_phrase():
    assert set(_PHRASE_PER_CLASS) == set(CLASS_NAMES)            # a phrase for all 16
    for cls, phrase in _PHRASE_PER_CLASS.items():
        assert event_type_from_text(phrase) == cls


def test_sixteen_classes():
    assert len(CLASS_NAMES) == 16
    assert len(set(CLASS_NAMES)) == 16


def test_enum_matches_tuple():
    assert tuple(e.value for e in EventType) == CLASS_NAMES


def test_false_positive_only_normal_sinus():
    assert is_false_positive("NORMAL_SINUS")
    for name in CLASS_NAMES:
        if name != "NORMAL_SINUS":
            assert not is_false_positive(name)


def test_validity():
    assert is_valid_event_type("ATRIAL_FIBRILLATION")
    assert not is_valid_event_type("NOT_A_CLASS")


def test_condition_codes_map_to_valid_names():
    assert len(CONDITION_CODE_TO_NAME) == 16
    assert set(CONDITION_CODE_TO_NAME.values()) == set(CLASS_NAMES)
    assert condition_code_to_name("AFIB") == "ATRIAL_FIBRILLATION"
    assert condition_code_to_name("N") == "NORMAL_SINUS"
    assert condition_code_to_name("UNKNOWN") is None


@pytest.mark.ecgtranscnn
def test_parity_with_upstream():
    pytest.importorskip("ecg_transcovnet")
    assert load_upstream_class_names() == list(CLASS_NAMES)


@pytest.mark.ecgtranscnn
def test_condition_code_parity_with_upstream():
    pytest.importorskip("ecg_transcovnet")
    assert load_upstream_condition_codes() == CONDITION_CODE_TO_NAME
