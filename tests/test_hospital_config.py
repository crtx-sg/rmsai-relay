"""Per-hospital config (`common/hospital_config.py` + `Config.from_env`): layering and validation.

Precedence: config/hospitals/default.yaml < <HOSPITAL_ID>.yaml < environment (logged override).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Config
from common.hospital_config import SETTINGS, hospital_settings, load_hospital_config

_HOSPITALS = Path(__file__).resolve().parents[1] / "config" / "hospitals"

_SITE = """
outbound:
  dispatch_mode: app
  call_number: "+15551234567"
  from: "+15557654321"
  inbound_allowed_numbers: ["+15550001111"]
escalation:
  mews_threshold: 4
"""


@pytest.fixture
def hospitals(tmp_path, monkeypatch):
    (tmp_path / "default.yaml").write_text((_HOSPITALS / "default.yaml").read_text(encoding="utf-8"),
                                           encoding="utf-8")
    (tmp_path / "h1.yaml").write_text(_SITE, encoding="utf-8")
    monkeypatch.setenv("HOSPITAL_CONFIG_DIR", str(tmp_path))
    for _field, (_s, _k, env) in SETTINGS.items():
        monkeypatch.delenv(env, raising=False)
    return tmp_path


def test_default_yaml_matches_the_code_defaults():
    """default.yaml replaces the old in-code defaults; it must not change behaviour on its own."""
    raw, source = load_hospital_config("", _HOSPITALS)
    plain = Config()
    assert source == "default.yaml"
    for field_name, value in hospital_settings(raw).items():
        assert getattr(plain, field_name) == value, field_name


def test_committed_example_is_valid(hospitals):
    """A site starts by copying h1.example.yaml to h1.yaml; the template must load as-is."""
    (hospitals / "h1.yaml").write_text((_HOSPITALS / "h1.example.yaml").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    raw, _ = load_hospital_config("h1", hospitals)
    assert hospital_settings(raw)["outbound_call_number"] == "+15551234567"


def test_hospital_file_overrides_the_default(hospitals, monkeypatch):
    monkeypatch.setenv("HOSPITAL_ID", "h1")
    c = Config.from_env()
    assert c.hospital_config_source == "default.yaml + h1.yaml"
    assert (c.dispatch_mode, c.outbound_call_number, c.outbound_from) == (
        "app", "+15551234567", "+15557654321")
    assert c.sip_inbound_allowed_numbers == ("+15550001111",)
    assert c.criticality_mews_threshold == 4
    assert c.outbound_max_retries == 2 and c.criticality_normal_event == "NORMAL_SINUS"  # defaults


def test_environment_overrides_the_hospital_file_and_is_logged(hospitals, monkeypatch, capsys):
    monkeypatch.setenv("HOSPITAL_ID", "h1")
    monkeypatch.setenv("OUTBOUND_CALL_NUMBER", "+15559998888")
    monkeypatch.setenv("CRITICALITY_MEWS_THRESHOLD", "5")
    c = Config.from_env()
    assert c.outbound_call_number == "+15559998888" and c.criticality_mews_threshold == 5
    err = capsys.readouterr().err
    assert "environment overrides hospital config" in err
    assert "OUTBOUND_CALL_NUMBER" in err and "+15559998888" not in err  # names only, never values


def test_no_hospital_id_uses_default_only(hospitals, monkeypatch):
    monkeypatch.delenv("HOSPITAL_ID", raising=False)
    c = Config.from_env()
    assert c.hospital_config_source == "default.yaml" and c.outbound_call_number == ""


@pytest.mark.parametrize("site, message", [
    ('outbound:\n  call_number: +15551234567\n', "quoted"),        # YAML reads it as an int
    ('outbound:\n  call_number: "5551234567"\n', "E.164"),
    ('outbound:\n  dispatch_mode: phone\n', "dispatch_mode"),
    ('outbound:\n  min_criticality: Urgent\n', "min_criticality"),
    ('escalation:\n  low_confidence_caveat: 1.5\n', "between 0 and 1"),
    ('outbound:\n  inbound_allowed_numbers: ["+1"]\n', "E.164"),
])
def test_invalid_site_file_fails_loudly(hospitals, site, message):
    (hospitals / "h1.yaml").write_text(site, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_hospital_config("h1", hospitals)


def test_missing_setting_in_default_is_reported(tmp_path):
    (tmp_path / "default.yaml").write_text("escalation: {}\noutbound: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing escalation.normal_event"):
        load_hospital_config("", tmp_path)


def test_vitals_alerts_do_not_call_by_default(monkeypatch):
    from common.config import Config

    monkeypatch.delenv("OUTBOUND_CALL_VITALS_ALERTS", raising=False)
    assert Config.from_env().outbound_call_vitals_alerts is False
    monkeypatch.setenv("OUTBOUND_CALL_VITALS_ALERTS", "true")
    assert Config.from_env().outbound_call_vitals_alerts is True
