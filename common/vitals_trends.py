"""Clinical-significance policy for vital trends, configurable per hospital.

The vendored Mann-Kendall test (`ecgtranscnn.mews.mann_kendall`) answers only "is there a consistent
monotonic drift?". With 20-30 readings a drift of 1-2 breaths/min is "highly significant", and the
vendored direction rule is fixed per vital (HR rising = bad, falling = good — so a worsening
bradycardia reads as "improving"). This module adds the clinical layer on top of its p-value:

1. consistent:  p < `alpha`
2. big enough:  |change over the window| >= the vital's `min_change` (time-based Sen's slope x span)
3. abnormal:    the latest reading is outside the vital's `normal` range, and moving away from it

Only then is the vital `deteriorating`. Back toward normal by >= `min_change` is `improving`;
anything else is `stable`, with a `reason` code saying which test it failed.

Policy lives in `config/vitals_trends/default.yaml`, overridden key-by-key by
`config/vitals_trends/<hospital_id>.yaml` when that file exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = _REPO_ROOT / "config" / "vitals_trends"
_HOSPITAL_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # the id becomes a file name: no paths

# `reason` codes on a classified trend
NOT_SIGNIFICANT = "not_significant"    # p >= alpha: no consistent drift
BELOW_MIN_CHANGE = "below_min_change"  # consistent, but smaller than the hospital's threshold
WITHIN_NORMAL = "within_normal"        # latest reading inside the normal range
AWAY_FROM_NORMAL = "away_from_normal"  # deteriorating
TOWARD_NORMAL = "toward_normal"        # improving


@dataclass(frozen=True)
class VitalRule:
    min_change: float
    normal_low: float
    normal_high: float
    unit: str = ""


@dataclass(frozen=True)
class TrendPolicy:
    alpha: float
    vitals: dict[str, VitalRule]
    source: str  # which files it was built from, for logs / the app


@dataclass(frozen=True)
class TrendVerdict:
    direction: str  # TrendDirection
    reason: str
    change: float | None = None    # signed, in the vital's units, over the window
    span_s: float | None = None    # first -> last reading, seconds
    latest: float | None = None


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_trend_policy(hospital_id: str = "", directory: str | Path | None = None) -> TrendPolicy:
    """`default.yaml`, then `<hospital_id>.yaml` on top (if present). Raises ValueError on a bad
    file or hospital id — a misconfigured alerting policy should stop startup, not run silently."""
    d = Path(directory) if directory else DEFAULT_DIR
    if not d.is_absolute():
        d = _REPO_ROOT / d
    raw = _read(d / "default.yaml")
    sources = ["default.yaml"]
    if hospital_id:
        if not _HOSPITAL_ID.match(hospital_id):
            raise ValueError(f"invalid hospital id for a trend policy file: {hospital_id!r}")
        site = d / f"{hospital_id}.yaml"
        if site.exists():
            raw = _merge(raw, _read(site))
            sources.append(site.name)

    alpha = float(raw.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError(f"vitals_trends alpha must be in (0, 1), got {alpha}")
    vitals = {}
    for name, r in (raw.get("vitals") or {}).items():
        try:
            low, high = (float(x) for x in r["normal"])
            rule = VitalRule(min_change=float(r["min_change"]), normal_low=low, normal_high=high,
                             unit=str(r.get("unit", "")))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"vitals_trends {name}: needs min_change and normal: [low, high] ({e})")
        if rule.min_change <= 0 or rule.normal_low >= rule.normal_high:
            raise ValueError(f"vitals_trends {name}: min_change must be > 0 and low < high")
        vitals[name] = rule
    return TrendPolicy(alpha=alpha, vitals=vitals, source=" + ".join(sources))


def _change_over_window(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """(change, span_s): Sen's slope in units per *second* (readings arrive at uneven intervals)
    times the window span. Robust to a single outlier, unlike last - first."""
    span = samples[-1][0] - samples[0][0]
    slopes = [(vj - vi) / (tj - ti)
              for i, (ti, vi) in enumerate(samples) for tj, vj in samples[i + 1:] if tj > ti]
    if span <= 0 or not slopes:
        return samples[-1][1] - samples[0][1], 0.0
    return median(slopes) * span, span


def classify_trend(samples: list[tuple[float, float]], p: float | None,
                   rule: VitalRule, alpha: float) -> TrendVerdict:
    """Classify one vital from its (timestamp, value) history, oldest first, and the Mann-Kendall
    p-value computed on those values."""
    if len(samples) < 3 or p is None:
        return TrendVerdict("insufficient_data", NOT_SIGNIFICANT)
    change, span = _change_over_window(samples)
    latest = samples[-1][1]
    v = dict(change=round(change, 2), span_s=span, latest=latest)
    if p >= alpha:
        return TrendVerdict("stable", NOT_SIGNIFICANT, **v)
    if abs(change) < rule.min_change:
        return TrendVerdict("stable", BELOW_MIN_CHANGE, **v)
    rising = change > 0
    if latest > rule.normal_high:
        away = rising
    elif latest < rule.normal_low:
        away = not rising
    else:
        # Inside the normal range now: never deteriorating. Improving if it started outside it.
        start = latest - change
        came_back = start > rule.normal_high or start < rule.normal_low
        return TrendVerdict("improving" if came_back else "stable",
                            TOWARD_NORMAL if came_back else WITHIN_NORMAL, **v)
    return TrendVerdict("deteriorating" if away else "improving",
                        AWAY_FROM_NORMAL if away else TOWARD_NORMAL, **v)
