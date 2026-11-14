"""`load_protocol_file` makes the graph's protocols exactly the hospital's file (no Neo4j needed)."""

from __future__ import annotations

import pytest

from kb.graph.protocols import load_protocol_file

_FILE = """
- id: vt_site
  title: Site VT protocol
  match: {event_type: VENTRICULAR_TACHYCARDIA}
  steps:
    - {order: 1, kind: test, text: 12-lead ECG}
    - {order: 2, kind: escalation, text: Call the rapid response team}
"""


class _Recorder:
    def __init__(self):
        self.calls = []

    def run_write(self, cypher, **params):
        self.calls.append((cypher, params))
        return []


def test_reload_deletes_protocols_and_steps_not_in_the_file(tmp_path):
    f = tmp_path / "h2.protocols.yaml"
    f.write_text(_FILE, encoding="utf-8")
    d = _Recorder()
    assert load_protocol_file(d, f) == 1
    deletes = [(c, p) for c, p in d.calls if "DELETE" in c]
    assert len(deletes) == 2 and deletes[0][1] == {"ids": ["vt_site"]}
    assert deletes[1][1] == {"sids": ["vt_site_step_1", "vt_site_step_2"]}
    # the deletes run after every MERGE, so a protocol in the file is never removed and re-added
    assert all("DELETE" not in c for c, _ in d.calls[:-2])


def test_an_empty_file_deletes_nothing(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("[]\n", encoding="utf-8")
    d = _Recorder()
    with pytest.raises(ValueError, match="no protocols"):
        load_protocol_file(d, f)
    assert d.calls == []
