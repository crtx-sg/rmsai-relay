"""E3: provenance, model identity, persisted decision/why/outcome, and the ground-truth link fix.

All offline: a recording fake driver stands in for Neo4j (the live graph is never touched).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import pytest

from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from inference.pipeline import process_window
from inference.serialize import dict_to_event, event_to_dict
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from kb.graph import events as graph_events

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


class FakeDriver:
    def __init__(self, rows=None):
        self.writes: list[tuple[str, dict]] = []
        self.reads: list[tuple[str, dict]] = []
        self._rows = rows or []

    def run_write(self, query, **params):
        self.writes.append((query, params))

    def run_read(self, query, **params):
        self.reads.append((query, params))
        return self._rows


def _ecg_sigma_file(tmp_path) -> Path:
    """The fixture, re-stamped the way ecg_sigma + cli.real_samples write a real-ECG file."""
    f = tmp_path / "real.h5"
    shutil.copy(_FIXTURE, f)
    with h5py.File(f, "r+") as h:
        md = h["metadata"]
        md.attrs.update({"source_dataset": "mitbih", "record_id": "mitbih:207",
                         "subject_id": "207", "ecgpkg_subject_id": "mitbih:207",
                         "ecgpkg_version": "v2", "ecgpkg_split": "test"})
        ev = h[next(k for k in h if k.startswith("event_"))]
        ev.attrs.update({"source_sample": 412800, "label_method": "rhythm_annotation",
                         "label_purity": 0.93, "source_condition": "VENTRICULAR_FLUTTER"})
    return f


# --- provenance -----------------------------------------------------------------------------

def test_provenance_ecg_sigma(tmp_path):
    p = next(read_hdf5_file(_ecg_sigma_file(tmp_path))).provenance
    assert p.model_dump(exclude_none=True) == {
        "kind": "ecg_sigma", "device": "RMSAI-SimDevice-v2.0", "dataset": "mitbih",
        "record": "mitbih:207", "subject": "mitbih:207", "source_sample": 412800,
        "label_method": "rhythm_annotation", "label_purity": 0.93,
        "source_label": "VENTRICULAR_FLUTTER", "package": "v2", "split": "test"}


def test_provenance_simulator_and_device(tmp_path):
    assert next(read_hdf5_file(_FIXTURE)).provenance.kind == "simulator"
    f = tmp_path / "dev.h5"
    shutil.copy(_FIXTURE, f)
    with h5py.File(f, "r+") as h:
        del h["metadata"]["device_info"]
        h["metadata"].create_dataset("device_info", data="Philips-IntelliVue-MX800")
    p = next(read_hdf5_file(f)).provenance
    assert (p.kind, p.device, p.dataset) == ("device", "Philips-IntelliVue-MX800", None)


# --- model identity + bus ---------------------------------------------------------------------

def test_model_identity_stamped_and_carried_on_the_bus(tmp_path):
    w = next(read_hdf5_file(_ecg_sigma_file(tmp_path)))
    ev = process_window(w, StubECGModel(), MewsVitalsAnalysis())
    assert ev.model_id == "stub" and ev.model_classes == list(CLASS_NAMES)
    back = dict_to_event(event_to_dict(ev))
    assert back.model_id == "stub" and back.model_classes == list(CLASS_NAMES)
    assert back.window.provenance == ev.window.provenance


def test_old_payload_without_new_fields_still_parses():
    ev = process_window(next(read_hdf5_file(_FIXTURE)), StubECGModel(), MewsVitalsAnalysis())
    payload = event_to_dict(ev)
    for k in ("model_id", "model_classes", "provenance"):
        payload.pop(k)
    old = dict_to_event(payload)
    assert old.model_id is None and old.window.provenance is None


# --- what gets persisted ----------------------------------------------------------------------

def test_traceability_props(tmp_path):
    from orchestrator.event_flow import traceability_props

    ev = process_window(next(read_hdf5_file(_ecg_sigma_file(tmp_path))), StubECGModel(),
                        MewsVitalsAnalysis())
    props = traceability_props(ev)
    assert props["model_id"] == "stub"
    assert (props["source_kind"], props["source_dataset"], props["source_record"],
            props["source_sample"], props["source_split"]) == (
        "ecg_sigma", "mitbih", "mitbih:207", 412800, "test")
    assert props["eval_outcome"] in {"TP", "TP_WRONG_CLASS", "FP", "FN", "TN"}
    assert props["why"].startswith(("Shown because", "Not alerted"))
    assert isinstance(props["alert_gate"], bool) and props["alert_reason_code"]
    assert json.loads(props["why_json"])["headline"] == props["why"]


def test_persist_sets_extra_props_and_drops_nones():
    d = FakeDriver()
    graph_events.persist_monitored_event(
        d, uuid="e1", patient_id="PT1", timestamp=1.0, event_type="PVC", confidence=0.9,
        is_false_positive=False, extra={"model_id": "stub", "source_dataset": None})
    extra_writes = [p for q, p in d.writes if "SET e += $props" in q]
    assert extra_writes == [{"uuid": "e1", "props": {"model_id": "stub"}}]


def test_condition_link_is_the_prediction_never_the_ground_truth():
    d = FakeDriver()
    graph_events.persist_monitored_event(
        d, uuid="e1", patient_id="PT1", timestamp=1.0, event_type="PVC", confidence=0.9,
        is_false_positive=False, ground_truth_condition="VENTRICULAR_TACHYCARDIA",
        link_condition="PVC")
    links = [p for q, p in d.writes if "OF_CONDITION" in q]
    assert [p["name"] for p in links] == ["PVC"]
    # without a link_condition, ground truth no longer stands in for one
    d2 = FakeDriver()
    graph_events.persist_monitored_event(
        d2, uuid="e2", patient_id="PT1", timestamp=1.0, event_type="PVC", confidence=0.9,
        is_false_positive=False, ground_truth_condition="VENTRICULAR_TACHYCARDIA")
    assert not [q for q, _ in d2.writes if "OF_CONDITION" in q]


def test_event_flow_links_the_prediction(monkeypatch):
    import orchestrator.event_flow as flow

    seen = {}

    class Stop(Exception):
        pass

    def capture(driver, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(flow, "persist_monitored_event", capture)
    ev = process_window(next(read_hdf5_file(_FIXTURE)), StubECGModel(), MewsVitalsAnalysis())
    with pytest.raises(Stop):
        flow.process_device_event(ev, FakeDriver(), vector=None)
    assert seen["link_condition"] == ev.event_type
    assert seen["ground_truth_condition"] == ev.window.ground_truth.condition  # kept, eval-only
    assert seen["extra"]["model_id"] == "stub"


def test_delivery_and_eval_queries():
    d = FakeDriver()
    graph_events.set_event_delivery(d, "e1", delivered_app=True, delivered_call=None,
                                    delivered_sms="sms_delivered")
    assert d.writes == [("MATCH (e:MonitoredEvent {id:$uuid}) SET e += $props",
                         {"uuid": "e1", "props": {"delivered_app": True,
                                                  "delivered_sms": "sms_delivered"}})]
    d2 = FakeDriver(rows=[{"patient": "PT1", "id": "e1"}])
    rows = graph_events.eval_events(d2, dataset="mitbih")
    query, params = d2.reads[0]
    assert rows == [{"patient": "PT1", "id": "e1"}]
    assert params == {"since": None, "model": None, "dataset": "mitbih", "labelled": True}
    assert "e.eval_outcome AS eval_outcome" in query and "e.source_record AS source_record" in query
