"""Display precision for vitals: integers, except temperature to one decimal.

Stored values stay raw (trend tests and MEWS use them); this is only for what people read or hear,
so `sbp 110.115`, `SpO2 96.3793` and `RR 14.999999850000002` become `110`, `96` and `15`, while
`temp 99.2236` becomes `99.2`. Accepts every name the codebase uses for a vital (device keys
`HR`/`Temp`, graph keys `hr`/`temp`, MEWS names `Heart Rate`/`Temperature`).
"""

from __future__ import annotations


def is_temperature(name: str) -> bool:
    return "temp" in (name or "").lower()


def round_vital(name: str, value):
    """The value at display precision (a number), or the input unchanged if it isn't numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if is_temperature(name):
        return round(float(value), 1)
    return int(round(value))


def fmt_vital(name: str, value) -> str:
    """`round_vital` as text: `110`, `96`, `99.2`."""
    v = round_vital(name, value)
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)
