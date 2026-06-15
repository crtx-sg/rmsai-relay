"""Care-protocol loader skeleton (D19)."""

from __future__ import annotations

from pathlib import Path

from common.protocol_loader import find_default, load_protocols

_CONFIG = Path(__file__).resolve().parents[1] / "common" / "protocols" / "care_protocols.yaml"


def test_loads_sample_config():
    protocols = load_protocols(_CONFIG)
    ids = {p.id for p in protocols}
    assert "afib_rvr" in ids and "default" in ids


def test_match_block_parsed():
    afib = next(p for p in load_protocols(_CONFIG) if p.id == "afib_rvr")
    assert afib.match.event_type == "ATRIAL_FIBRILLATION"
    assert "HR > 120" in afib.match.vital_conditions
    assert afib.match.min_severity == "High"


def test_steps_have_kind_specific_fields():
    afib = next(p for p in load_protocols(_CONFIG) if p.id == "afib_rvr")
    med = next(s for s in afib.steps if s.kind == "medication")
    assert med.fields.get("route") == "IV"


def test_default_fallback_present():
    default = find_default(load_protocols(_CONFIG))
    assert default is not None and default.match.event_type == "*"


def test_render_produces_narrative():
    afib = next(p for p in load_protocols(_CONFIG) if p.id == "afib_rvr")
    text = afib.render()
    assert "Atrial fibrillation" in text
    assert "[medication]" in text
