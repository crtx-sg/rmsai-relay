"""event_type vocabulary: parity with upstream + helpers."""

from __future__ import annotations

import pytest

from common.event_types import (
    CLASS_NAMES,
    EventType,
    is_false_positive,
    is_valid_event_type,
    load_upstream_class_names,
)


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


@pytest.mark.ecgtranscnn
def test_parity_with_upstream():
    pytest.importorskip("ecg_transcovnet")
    assert load_upstream_class_names() == list(CLASS_NAMES)
