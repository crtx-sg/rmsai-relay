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

#: Maps the simulator's short ground-truth `condition` codes (HDF5 event attr, == the
#: `Condition` enum *values*) to our class names (== enum *names*). The reader uses this to
#: translate the sim-only ground truth; it is NEVER the source of the predicted event_type.
#: Mirrors `ecg_transcovnet.simulator.conditions.Condition`; parity-tested.
CONDITION_CODE_TO_NAME: dict[str, str] = {
    "N": "NORMAL_SINUS",
    "SB": "SINUS_BRADYCARDIA",
    "ST": "SINUS_TACHYCARDIA",
    "AFIB": "ATRIAL_FIBRILLATION",
    "AFL": "ATRIAL_FLUTTER",
    "A": "PAC",
    "SVTA": "SVT",
    "V": "PVC",
    "VT": "VENTRICULAR_TACHYCARDIA",
    "VF": "VENTRICULAR_FIBRILLATION",
    "L": "LBBB",
    "R": "RBBB",
    "1AVB": "AV_BLOCK_1",
    "2AVB1": "AV_BLOCK_2_TYPE1",
    "2AVB2": "AV_BLOCK_2_TYPE2",
    "STE": "ST_ELEVATION",
}


def condition_code_to_name(code: str) -> str | None:
    """Translate a sim ground-truth condition code (e.g. 'AFIB') to a class name."""
    return CONDITION_CODE_TO_NAME.get(code)


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


def load_upstream_condition_codes() -> dict[str, str]:
    """Return the upstream `{code: name}` map (for the parity test). Pulls torch — guard it."""
    from ecg_transcovnet.simulator.conditions import Condition  # noqa: PLC0415

    return {c.value: c.name for c in Condition}
