"""Rational sample-rate handling (decision E).

Device metadata stores RESP as the rounded float `33.33`, but the true rate is `100/3`.
Rounding it propagates sub-sample drift, so we snap known rounded device rates back to exact
fractions and otherwise derive a clean rational from the float.
"""

from __future__ import annotations

from fractions import Fraction

# Known rounded device rates -> exact rational (within a small tolerance of the stored float).
_KNOWN_RATES: tuple[tuple[float, Fraction], ...] = (
    (100 / 3, Fraction(100, 3)),  # RESP, stored as 33.33
)
_SNAP_TOL = 0.05


def to_rational_rate(stored: float) -> Fraction:
    """Return an exact `Fraction` for a stored (possibly rounded) sample rate."""
    for target_f, exact in _KNOWN_RATES:
        if abs(stored - target_f) < _SNAP_TOL:
            return exact
    # Integer-valued rates (200.0, 75.0) stay exact; others get a clean rational.
    if float(stored).is_integer():
        return Fraction(int(stored))
    return Fraction(stored).limit_denominator(1000)
