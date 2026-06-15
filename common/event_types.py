"""The 16-class `event_type` vocabulary.

Source of truth is `ecg_transcovnet.constants.CLASS_NAMES` (the simulator's `Condition`
enum). We mirror it here as a plain tuple so importing this module does **not** pull in
`torch`/`matplotlib` (which the upstream package `__init__` eagerly imports). A parity test
(`tests/test_event_types.py`) asserts this list matches upstream whenever the vendored package
is importable, so the two can never silently diverge.
"""

from __future__ import annotations

from enum import Enum

# Order matches `ecg_transcovnet.simulator.conditions.Condition` (== model output index order).
CLASS_NAMES: tuple[str, ...] = (
    "NORMAL_SINUS",
    "SINUS_BRADYCARDIA",
    "SINUS_TACHYCARDIA",
    "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER",
    "PAC",
    "SVT",
    "PVC",
    "VENTRICULAR_TACHYCARDIA",
    "VENTRICULAR_FIBRILLATION",
    "LBBB",
    "RBBB",
    "AV_BLOCK_1",
    "AV_BLOCK_2_TYPE1",
    "AV_BLOCK_2_TYPE2",
    "ST_ELEVATION",
)

#: Predicting this class means the event is a false positive (no real arrhythmia).
NORMAL_SINUS = "NORMAL_SINUS"


class EventType(str, Enum):
    """Typed enum over the 16 classes; `EventType.NORMAL_SINUS.value == "NORMAL_SINUS"`."""

    NORMAL_SINUS = "NORMAL_SINUS"
    SINUS_BRADYCARDIA = "SINUS_BRADYCARDIA"
    SINUS_TACHYCARDIA = "SINUS_TACHYCARDIA"
    ATRIAL_FIBRILLATION = "ATRIAL_FIBRILLATION"
    ATRIAL_FLUTTER = "ATRIAL_FLUTTER"
    PAC = "PAC"
    SVT = "SVT"
    PVC = "PVC"
    VENTRICULAR_TACHYCARDIA = "VENTRICULAR_TACHYCARDIA"
    VENTRICULAR_FIBRILLATION = "VENTRICULAR_FIBRILLATION"
    LBBB = "LBBB"
    RBBB = "RBBB"
    AV_BLOCK_1 = "AV_BLOCK_1"
    AV_BLOCK_2_TYPE1 = "AV_BLOCK_2_TYPE1"
    AV_BLOCK_2_TYPE2 = "AV_BLOCK_2_TYPE2"
    ST_ELEVATION = "ST_ELEVATION"


def is_valid_event_type(event_type: str) -> bool:
    """True iff `event_type` is one of the 16 known classes."""
    return event_type in CLASS_NAMES


def is_false_positive(event_type: str) -> bool:
    """A `NORMAL_SINUS` prediction is, by definition, a false positive."""
    return event_type == NORMAL_SINUS


def load_upstream_class_names() -> list[str]:
    """Return `CLASS_NAMES` from the vendored ecgtranscnn package (for the parity test).

    Imports the package, which pulls torch/matplotlib — callers should guard with
    `pytest.importorskip` when the vendored tree may be absent.
    """
    from ecg_transcovnet.constants import CLASS_NAMES as upstream  # noqa: PLC0415

    return list(upstream)
