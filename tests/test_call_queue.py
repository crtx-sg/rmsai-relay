"""Call queue: severity order, restart recovery, and the app never waiting on a phone call."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from inference.serialize import event_to_dict
from orchestrator import bus_consumer
from orchestrator.call_queue import CallDispatcher, CallJob, CallQueue
from tests.test_bus_consumer import _Afib, _event, _FakeAlertStore, _run, patched  # noqa: F401


class _FakeRedis:
    """Just the sorted-set, hash and key commands the queue uses."""

    def __init__(self):
        self.z: dict[str, dict[str, float]] = {}
        self.h: dict[str, dict[str, str]] = {}
        self.kv: dict[str, str] = {}

    def hsetnx(self, key, field, value):
        h = self.h.setdefault(key, {})
        if field in h:
            return 0
        h[field] = value
        return 1

    def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    def hdel(self, key, field):
        return int(self.h.get(key, {}).pop(field, None) is not None)

    def hgetall(self, key):
        return dict(self.h.get(key, {}))

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)

    def zpopmin(self, key, count=1):
        z = self.z.get(key, {})
        out = sorted(z.items(), key=lambda kv: kv[1])[:count]
        for member, _ in out:
            del z[member]
        return out

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zrange(self, key, start, end):
        return [m for m, _ in sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])]

    # plain keys (the call slot); expiry is simulated by deleting the key in a test
    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, key):
        self.kv.pop(key, None)

    def eval(self, script, numkeys, key, value, *args):  # the queue's two compare-and-act scripts
        if self.kv.get(key) != value:
            return 0
        if "'del'" in script:
            del self.kv[key]
        return 1


def _job(event_id, criticality, ms):
    return CallJob(event_id=event_id, patient_ref="PT1", event_type="VENTRICULAR_FIBRILLATION",
                   bed="Unit1/Bed1", criticality=criticality, reason="ok", payload={},
                   enqueued_ms=ms)


def test_pops_highest_criticality_first_then_fifo():
    q = CallQueue(_FakeRedis())
    q.push(_job("high-1", "High", 1000))
    q.push(_job("crit-late", "Critical", 5000))
    q.push(_job("high-2", "High", 2000))
    q.push(_job("crit-early", "Critical", 3000))
    assert [q.pop().event_id for _ in range(4)] == ["crit-early", "crit-late", "high-1", "high-2"]
    assert q.pop() is None


def test_redelivered_event_keeps_its_place():
    q = CallQueue(_FakeRedis())
    assert q.push(_job("a", "High", 1000)) == 1
    assert q.push(_job("a", "Critical", 9000)) == 1  # same event again: not queued twice
    assert q.pop().criticality == "High"


def test_unfinished_call_is_recovered_after_a_crash():
    r = _FakeRedis()
    q = CallQueue(r)
    q.push(_job("a", "Critical", 1000))
    q.push(_job("b", "High", 2000))
    assert q.pop().event_id == "a"  # dialling "a" when the process died: no done()
    q2 = CallQueue(r)               # the restarted consumer
    assert q2.recover() == 1
    assert [q2.pop().event_id, q2.pop().event_id] == ["a", "b"]


def test_finished_call_is_not_recovered():
    q = CallQueue(_FakeRedis())
    q.push(_job("a", "High", 1000))
    q.done(q.pop().event_id)
    assert q.recover() == 0 and q.pop() is None


def test_second_consumer_neither_requeues_nor_dials_a_call_in_progress():
    r = _FakeRedis()
    q1, q2 = CallQueue(r), CallQueue(r)
    q1.push(_job("a", "Critical", 1000))
    q1.push(_job("b", "High", 2000))
    assert q1.claim().event_id == "a"   # consumer 1 is dialling "a"
    assert q2.recover() == 0            # consumer 2 starts: "a" is not an orphan
    assert q2.claim() is None           # ...and it may not ring a second phone meanwhile
    q1.done("a")
    assert q2.claim().event_id == "b"


def test_crashed_holder_job_is_recovered_once_its_lease_lapses():
    r = _FakeRedis()
    q1, q2 = CallQueue(r), CallQueue(r)
    q1.push(_job("a", "Critical", 1000))
    q1.push(_job("b", "High", 2000))
    assert q1.claim().event_id == "a"   # consumer 1 dies mid-call: no done(), no keep_alive()
    r.delete(q1.active_key)             # its lease expires
    assert q2.claim().event_id == "a"   # the orphan is re-queued, most severe first


def test_dispatcher_places_calls_one_at_a_time_and_survives_a_failing_one():
    q = CallQueue(_FakeRedis())
    for i, c in enumerate(["High", "Critical", "High"]):
        q.push(_job(f"e{i}", c, 1000 + i))
    active, peak, placed = [0], [0], []

    def _handle(job):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        placed.append(job.event_id)
        if job.event_id == "e0":
            raise RuntimeError("carrier down")

    d = CallDispatcher(q, _handle, poll_s=0.05)
    d.start()
    d.stop(drain=True, timeout=5)
    assert placed == ["e1", "e0", "e2"] and peak[0] == 1
    assert q.recover() == 0  # the failed job was dropped, not left to redial forever


# --- bus consumer integration ----------------------------------------------------------------------

def test_queued_call_returns_before_dialling(patched):  # noqa: F811
    q = CallQueue(_FakeRedis())
    store = _FakeAlertStore()
    res = _run(event_to_dict(_event(_Afib())), patched, channel="voice",
               caller_factory=lambda room: object(), dispatch_fn=lambda room: None,
               alert_store=store, call_end_waiter=lambda room: 0.0, call_queue=q)
    assert res.call_queued and not res.called and res.queue_depth == 1
    assert patched["voice"] == [] and store.staged == []  # nothing dialled or staged yet
    assert q.pop().event_id == res.event_uuid


def test_app_gets_every_event_while_a_call_is_in_progress(patched, monkeypatch):  # noqa: F811
    # The demo complaint: the worklist only showed event N+1 once call N hung up.
    hang_up = threading.Event()
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: "reported")
    q = CallQueue(_FakeRedis())
    pushed = []

    class _Inbox:
        room = "rmsai-inbox-test"

        def publish_event(self, message):
            pushed.append(message)

    kw = dict(driver=object(), orchestrator=object(), utterances=["yes"],
              config=bus_consumer.DEFAULT, caller_factory=lambda room: object(),
              dispatch_fn=lambda room: None, alert_store=_FakeAlertStore(),
              call_end_waiter=lambda room: hang_up.wait(5) and 1.0)
    d = CallDispatcher(q, lambda job: bus_consumer.run_queued_call(job, **kw), poll_s=0.02)
    d.start()
    cfg = replace(bus_consumer.DEFAULT, outbound_enabled=True, dispatch_mode="app+call",
                  outbound_min_criticality="High")
    payloads = []
    for _ in range(3):
        p = event_to_dict(_event(_Afib()))
        p["event_id"] = f"evt-{len(payloads)}"
        payloads.append(p)
    for p in payloads:
        _run(p, patched, config=cfg, channel="voice", caller_factory=kw["caller_factory"],
             dispatch_fn=kw["dispatch_fn"], alert_store=kw["alert_store"],
             call_end_waiter=kw["call_end_waiter"], call_queue=q, inbox_publisher=_Inbox())
        d.notify()
    time.sleep(0.2)
    assert len(pushed) == 3          # all three on the worklist...
    assert len(patched["voice"]) == 1  # ...while the first call is still up
    hang_up.set()
    d.stop(drain=True, timeout=5)
    assert len(patched["voice"]) == 3


def test_queued_call_is_skipped_once_acknowledged(patched, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: "acknowledged")
    job = CallJob(event_id="e", patient_ref="PT1", event_type="ATRIAL_FIBRILLATION", bed="ICU/3",
                  criticality="High", reason="ok", payload=event_to_dict(_event(_Afib())))
    assert bus_consumer.run_queued_call(
        job, driver=object(), orchestrator=object(), utterances=[], caller_factory=lambda r: object(),
        alert_store=_FakeAlertStore()) is None
    assert patched["voice"] == []


def test_status_lookup_failure_still_calls(patched, monkeypatch):  # noqa: F811
    def _boom(d, uuid):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(bus_consumer, "get_event_status", _boom)
    monkeypatch.setattr(bus_consumer, "_record_delivery", lambda *a, **k: None)
    job = CallJob(event_id="e", patient_ref="PT1", event_type="ATRIAL_FIBRILLATION", bed="ICU/3",
                  criticality="High", reason="ok", payload=event_to_dict(_event(_Afib())))
    store = _FakeAlertStore()
    res = bus_consumer.run_queued_call(
        job, driver=object(), orchestrator=object(), utterances=[], caller_factory=lambda r: object(),
        alert_store=store)
    assert res.called and store.staged  # staged at dial time
    assert patched["voice"] == ["ICU/3"]
