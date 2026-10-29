"""Live call 2026-10-29: "bed? 23" and "bed twenty-three" missed the bed; deteriorating vitals for a
bed answered for the called patient; "patients assigned to bed 23" read out the whole ward."""
from __future__ import annotations

import pytest

from kb.graph.lookup import match_intent
from kb.graph.spoken import normalize_spoken_query
from voice.outbound import parse_ack


@pytest.mark.parametrize("said", ["on bed? 23", "on bed twenty-three", "on bed twenty three",
                                  "on bed two three", "on bed 23"])
def test_bed_twenty_three_in_every_spoken_form(said):
    assert normalize_spoken_query(said) == "on bed Unit1-Bed23"


def test_tens_and_old_forms_unchanged():
    assert normalize_spoken_query("bed twenty") == "bed Unit1-Bed20"
    assert normalize_spoken_query("unit one bed oh one") == "bed Unit1-Bed01"
    assert normalize_spoken_query("which bed? is it") == "which bed? is it"


def _route(q):
    return match_intent(q, now=0.0, patient_ref="PT1", event_ref="E1")


def test_deteriorating_vitals_for_a_bed_are_that_beds():
    assert _route("Are there any deteriorating vitals for the patient on bed? 23") == (
        "bed_last_event", {"bed": "Unit1-Bed23", "focus": "trends"})
    assert _route("deteriorating vitals for this patient") == (
        "event_summary", {"event_uuid": "E1", "focus": "trends"})


@pytest.mark.parametrize("q", ["Are there any patients assigned to bed twenty-three?",
                               "Is any patient occupied bed twenty-three?", "who is on bed 23"])
def test_who_is_on_a_bed(q):
    assert _route(q) == ("bed_last_event", {"bed": "Unit1-Bed23"})


def test_acknowledged_plus_one_misheard_word_is_an_ack():
    assert parse_ack("Acknowledged, Emmett") == "yes"
    assert parse_ack("Acknowledge the Emmett") == "yes"
    assert parse_ack("Acknowledged, but not yet") == "no"
    assert parse_ack("Acknowledge what potassium level") == "unclear"
    # Another word after "acknowledge" is a different request, not a mishearing.
    for said in ("acknowledge later", "acknowledge what", "acknowledge pending"):
        assert parse_ack(said) == "unclear", said


def test_plural_pvcs_and_pacs_name_the_rhythm():
    from common.event_types import event_type_from_text

    assert event_type_from_text("What is the protocol for managing PACs?") == "PAC"
    assert event_type_from_text("any PVCs on bed 5") == "PVC"


def test_spoken_beds_name_the_unit_only_across_units():
    from orchestrator.orchestrator import _spoken_beds

    assert _spoken_beds(["Unit1-Bed02", "Unit1-Bed04"]) == "bed 2, bed 4"
    assert _spoken_beds(["Unit1-Bed02", "Unit2-Bed02"]) == "unit 1 bed 2, unit 2 bed 2"
