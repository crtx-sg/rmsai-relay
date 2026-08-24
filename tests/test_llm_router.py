"""LLM fallback routing — the validation layer that lets a model's choice reach Cypher.

The model is allowed to pick a NAME from a fixed menu and nothing else. Everything here is about
what happens to output that is wrong, malformed, or dangerous: an invented patient id must never
become a query against another patient's chart, and unusable output must degrade to "fall through
to document retrieval", never to an error.
"""

from __future__ import annotations

from kb.graph.llm_router import (
    CATALOGUE,
    build_prompt,
    looks_operational,
    parse_choice,
    resolve,
    route,
)

NOW = 1_700_000_000.0


class _Fake:
    """An LLM that says whatever it was constructed with (or explodes, to test the guard)."""

    def __init__(self, reply="", boom=False):
        self.reply, self.boom, self.prompts = reply, boom, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.boom:
            raise RuntimeError("ollama is down")
        return self.reply


# --- the cheap gate -----------------------------------------------------------------------------

def test_only_operational_questions_reach_the_model():
    # The router costs seconds on a local model. A clinical-literature question must not pay for it.
    assert looks_operational("how is the patient in bed three doing?")
    assert looks_operational("any critical events overnight?")
    assert not looks_operational("what is the SOP for handling atrial fibrillation?")
    assert not looks_operational("which anticoagulant is preferred?")


def test_a_document_question_never_calls_the_llm():
    llm = _Fake('{"name": "cohort_patterns"}')
    assert route("what is the sepsis protocol?", llm, now=NOW) is None
    assert llm.prompts == []  # not merely ignored — never invoked


# --- parsing what a small model actually emits ---------------------------------------------------

def test_json_is_recovered_from_surrounding_prose():
    # llama-class models wrap JSON in chatter and code fences; refusing to parse that would make the
    # router useless in exactly the deployment it is for.
    assert parse_choice('Sure! ```json\n{"name": "cohort_patterns"}\n```') == ("cohort_patterns", {})


def test_unusable_output_is_a_no_match_not_a_crash():
    for raw in ("", "I don't know", "{oops", '{"name": null}', '{"params": {}}', None):
        assert parse_choice(raw) is None


# --- validation: the layer that makes this safe --------------------------------------------------

def test_an_invented_patient_id_can_never_reach_a_query():
    # The whole safety argument: a model that hallucinates PT9999 would otherwise pull a different
    # patient's chart. Identity comes from session scope; the model's own params are discarded.
    choice = ("vitals_at_patient_last_event", {"patient_id": "PT9999"})
    assert resolve(choice, patient_ref="PT1155", now=NOW) == (
        "vitals_at_patient_last_event", {"patient_id": "PT1155"})
    # ...and with nobody in scope it refuses outright rather than guessing.
    assert resolve(choice, patient_ref=None, now=NOW) is None


def test_the_selected_event_comes_from_scope_too():
    choice = ("vitals_at_selected_event", {"event_uuid": "made-up"})
    assert resolve(choice, event_ref="evt-1", now=NOW) == (
        "vitals_at_selected_event", {"event_uuid": "evt-1"})
    assert resolve(choice, event_ref=None, now=NOW) is None


def test_a_template_outside_the_catalogue_is_refused():
    assert resolve(("drop_everything", {}), now=NOW) is None
    assert resolve(("vitals_at_event", {"event_uuid": "x"}), now=NOW) is None  # real, but not routable


def test_rhythm_types_are_enum_checked():
    ok = resolve(("patients_with_event_type", {"event_type": "atrial fibrillation"}), now=NOW)
    assert ok == ("patients_with_event_type", {"event_type": "ATRIAL_FIBRILLATION"})
    assert resolve(("patients_with_event_type", {"event_type": "FLUTTERY_HEART"}), now=NOW) is None


def test_bed_labels_must_look_like_bed_labels():
    assert resolve(("event_status_on_bed", {"bed": "Unit1-Bed01"}), now=NOW) == (
        "event_status_on_bed", {"bed": "Unit1-Bed01"})
    assert resolve(("event_status_on_bed", {"bed": "'; MATCH (n) DETACH DELETE n //"}), now=NOW) is None
    assert resolve(("event_status_on_bed", {"bed": ""}), now=NOW) is None


def test_time_windows_become_a_timestamp_and_reject_nonsense():
    assert resolve(("critical_events_since", {"hours": 6}), now=NOW) == (
        "critical_events_since", {"since": NOW - 6 * 3600})
    assert resolve(("positive_events_since", {"minutes": "30"}), now=NOW) == (
        "positive_events_since", {"since": NOW - 1800})
    for bad in ({"hours": "soon"}, {"hours": 0}, {"hours": -5}, {"hours": 10**9}, {}):
        assert resolve(("critical_events_since", bad), now=NOW) is None


def test_parameterless_templates_take_nothing_from_the_model():
    assert resolve(("outstanding_action_items", {"bed": "ignored"}), now=NOW) == (
        "outstanding_action_items", {})


# --- the prompt ----------------------------------------------------------------------------------

def test_scoped_options_are_hidden_when_there_is_no_scope():
    # Offering "this patient's events" with no patient in scope invites the model to invent one.
    without = build_prompt("how is bed 3?", has_patient=False, has_event=False)
    assert "vitals_at_patient_last_event" not in without
    assert "vitals_at_selected_event" not in without
    with_scope = build_prompt("how is my patient?", has_patient=True, has_event=True)
    assert "vitals_at_patient_last_event" in with_scope and "vitals_at_selected_event" in with_scope
    assert "event_status_on_bed" in without  # bed-scoped options need no session identity


def test_every_catalogue_entry_is_a_real_template():
    from kb.graph.templates import TEMPLATES

    assert set(CATALOGUE) <= set(TEMPLATES)


# --- end to end, with the model faked ------------------------------------------------------------

def test_a_missed_phrasing_routes_to_its_template():
    llm = _Fake('{"name": "event_status_on_bed", "params": {"bed": "Unit1-Bed01"}}')
    assert route("how is the patient in Unit1-Bed01 doing?", llm, now=NOW) == (
        "event_status_on_bed", {"bed": "Unit1-Bed01"})


def test_a_dead_llm_costs_the_turn_nothing_but_time():
    assert route("any critical events in the last 6 hours?", _Fake(boom=True), now=NOW) is None
