"""Per-hospital policy files: vital trends, escalation, outbound alerting.

`config/hospitals/default.yaml` holds every setting; `config/hospitals/<HOSPITAL_ID>.yaml`, when it
exists, overrides it key by key. An environment variable (the shell, or `.env`) overrides both: that
is the explicit escape hatch for a test, and `Config.from_env` logs every such override by name.

Site files hold real phone numbers, so they are gitignored; `<id>.example.yaml` is the committed
template. This module only reads and validates YAML; `common.config` maps the `escalation` and
`outbound` sections onto `Config`, `common.vitals_trends` reads `vitals_trends`.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "config/hospitals"
_HOSPITAL_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # the id becomes a file name: no paths

#: Config field -> (section, key, environment variable). The env var overrides the file.
SETTINGS: dict[str, tuple[str, str, str]] = {
    "criticality_normal_event": ("escalation", "normal_event", "CRITICALITY_NORMAL_EVENT"),
    "criticality_mews_threshold": ("escalation", "mews_threshold", "CRITICALITY_MEWS_THRESHOLD"),
    "criticality_escalate_on_deteriorating":
        ("escalation", "escalate_on_deteriorating", "CRITICALITY_ESCALATE_ON_DETERIORATING"),
    "criticality_fp_override_on_vitals":
        ("escalation", "fp_override_on_vitals", "CRITICALITY_FP_OVERRIDE_ON_VITALS"),
    "fp_suppress_min_confidence":
        ("escalation", "fp_suppress_min_confidence", "FP_SUPPRESS_MIN_CONFIDENCE"),
    "low_confidence_caveat": ("escalation", "low_confidence_caveat", "LOW_CONFIDENCE_CAVEAT"),
    "dispatch_mode": ("outbound", "dispatch_mode", "DISPATCH_MODE"),
    "outbound_enabled": ("outbound", "enabled", "OUTBOUND_ENABLED"),
    "outbound_call_number": ("outbound", "call_number", "OUTBOUND_CALL_NUMBER"),
    "outbound_from": ("outbound", "from", "OUTBOUND_FROM"),
    "outbound_min_criticality": ("outbound", "min_criticality", "OUTBOUND_MIN_CRITICALITY"),
    "outbound_min_arrhythmia_confidence":
        ("outbound", "min_arrhythmia_confidence", "OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE"),
    "outbound_call_vitals_alerts":
        ("outbound", "call_vitals_alerts", "OUTBOUND_CALL_VITALS_ALERTS"),
    "outbound_max_retries": ("outbound", "max_retries", "OUTBOUND_MAX_RETRIES"),
    "outbound_retry_delay_s": ("outbound", "retry_delay_s", "OUTBOUND_RETRY_DELAY_S"),
    "sip_inbound_allowed_numbers":
        ("outbound", "inbound_allowed_numbers", "SIP_INBOUND_ALLOWED_NUMBERS"),
}

_LEVELS = ("Low", "Medium", "High", "Critical")
_DISPATCH = ("app", "call", "app+call")
_PHONE = re.compile(r"^\+[1-9]\d{6,14}$")  # E.164


def resolve_dir(directory: str | Path | None) -> Path:
    d = Path(directory or DEFAULT_DIR)
    return d if d.is_absolute() else _REPO_ROOT / d


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


def load_hospital_config(hospital_id: str = "",
                         directory: str | Path | None = None) -> tuple[dict, str]:
    """`(merged settings, source)` from `default.yaml` + `<hospital_id>.yaml` (if present).

    Raises ValueError on a bad file or hospital id: a misconfigured alerting policy should stop
    startup, not run silently.
    """
    d = resolve_dir(directory)
    raw = _read(d / "default.yaml")
    sources = ["default.yaml"]
    if hospital_id:
        if not _HOSPITAL_ID.match(hospital_id):
            raise ValueError(f"invalid hospital id for a config file: {hospital_id!r}")
        site = d / f"{hospital_id}.yaml"
        if site.exists():
            raw = _merge(raw, _read(site))
            sources.append(site.name)
    source = " + ".join(sources)
    _validate(raw, source)
    return raw, source


def hospital_settings(raw: dict) -> dict:
    """The `escalation` / `outbound` values as `{Config field: value}`."""
    out = {}
    for field_name, (section, key, _env) in SETTINGS.items():
        out[field_name] = (raw.get(section) or {}).get(key)
    out["sip_inbound_allowed_numbers"] = tuple(out["sip_inbound_allowed_numbers"] or ())
    return out


def _validate(raw: dict, source: str) -> None:
    def bad(msg):
        raise ValueError(f"hospital config ({source}): {msg}")

    for field_name, (section, key, _env) in SETTINGS.items():
        if key not in (raw.get(section) or {}):
            bad(f"missing {section}.{key}")
    esc, out = raw["escalation"], raw["outbound"]
    for k in ("fp_suppress_min_confidence", "low_confidence_caveat"):
        if not 0 <= float(esc[k]) <= 1:
            bad(f"escalation.{k} must be between 0 and 1")
    if not 0 <= float(out["min_arrhythmia_confidence"]) <= 1:
        bad("outbound.min_arrhythmia_confidence must be between 0 and 1")
    if out["dispatch_mode"] not in _DISPATCH:
        bad(f"outbound.dispatch_mode must be one of {', '.join(_DISPATCH)}")
    if out["min_criticality"] not in _LEVELS:
        bad(f"outbound.min_criticality must be one of {', '.join(_LEVELS)}")
    numbers = [("outbound.call_number", out["call_number"]), ("outbound.from", out["from"])]
    numbers += [("outbound.inbound_allowed_numbers", n) for n in out["inbound_allowed_numbers"] or ()]
    for name, value in numbers:
        if value in ("", None):
            continue
        if not isinstance(value, str):  # an unquoted +91... is read by YAML as an integer
            bad(f"{name} must be a quoted string, e.g. \"+15551234567\"")
        if not _PHONE.match(value):
            bad(f"{name} must be an E.164 number like \"+15551234567\"")
