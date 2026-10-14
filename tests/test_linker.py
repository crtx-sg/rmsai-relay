"""Condition linking matches whole words only (offline: a fake driver returns the graph's names)."""

from __future__ import annotations

from kb.graph.entities import condition_id
from kb.hybrid.linker import link_conditions


class _FakeDriver:
    def __init__(self, names: list[str]) -> None:
        self.rows = [{"id": condition_id(n), "name": n} for n in names]

    def run_read(self, *_args, **_kwargs):
        return self.rows


def test_short_names_do_not_match_inside_words():
    driver = _FakeDriver(["MI", "CAD", "myocardial infarction"])
    assert link_conditions(driver, "what dose of amiodarone should i use") == []
    assert link_conditions(driver, "events in the last 30 minutes") == []
    assert link_conditions(driver, "decadent arcade") == []


def test_whole_word_still_links():
    driver = _FakeDriver(["myocardial infarction", "atrial fibrillation"])
    assert set(link_conditions(driver, "prior MI and afib?")) == {
        condition_id("myocardial infarction"), condition_id("atrial fibrillation")}
