"""Call queue: severity order, restart recovery, and the app never waiting on a phone call."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from inference.serialize import event_to_dict
from voice.outbound import SimulatedCaller
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

    def zscore(self, key, member):
        return self.z.get(key, {}).get(member)

    def zrangebyscore(self, key, lo, hi, start=0, num=None):
        top = float(hi[1:]) if str(hi).startswith("(") else float(hi)
        hits = [m for m, sc in sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1]) if sc < top]
        return hits[start:start + num] if num is not None else hits

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
        return int(self.kv.pop(key, None) is not None)

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


def test_crash_after_the_pin_does_not_ring_again():
    r = _FakeRedis()
    q1, q2 = CallQueue(r), CallQueue(r)
    q1.push(_job("a", "Critical", 1000))
    assert q1.claim().event_id == "a"
    q1.mark_answered("a")               # the clinician passed the PIN, then consumer 1 died
    r.delete(q1.active_key)             # its lease expires
    assert q2.recover() == 0 and q2.claim() is None
    assert r.hget(q2.jobs_key, "a") is None  # dropped, not left to be found again


def test_crash_before_the_pin_still_rings_again():
    r = _FakeRedis()
    q1, q2 = CallQueue(r), CallQueue(r)
    q1.push(_job("a", "Critical", 1000))
    assert q1.claim().event_id == "a"   # ringing, no PIN yet, when consumer 1 died
    r.delete(q1.active_key)
    assert q2.claim().event_id == "a"


def test_right_pin_marks_the_call_answered(tmp_path):
    from common.audit import AuditLog
    from tests.test_outbound_voice import _FakeOrch, _FakeWorking
    from voice.handlers import OutboundHandler
    from voice.outbound_alert import OutboundAlert

    answered = []
    h = OutboundHandler(_FakeOrch(), _FakeWorking(), OutboundAlert("room1", "PT1", "E-VF", "Alert"),
                        driver=object(), audit=AuditLog(tmp_path / "a.jsonl"),
                        on_answered=answered.append)
    h.respond("nine nine nine nine", session_id="room1")
    assert answered == []               # a wrong PIN is not an answer
    h.respond("one two three four", session_id="room1")
    assert answered == ["E-VF"]


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
              config=bus_consumer.DEFAULT, caller_factory=lambda room: SimulatedCaller(),
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


# --- 2026-11-13 (h2): a Critical VT waited 10+ min behind a High call's redials --------------------


def test_only_a_job_ahead_of_the_original_place_outranks():
    q = CallQueue(_FakeRedis())
    high = _job("high", "High", 1000)
    crit = _job("crit", "Critical", 1000)
    q.push(_job("crit-later", "Critical", 5000))
    assert q.outranked(high)  # higher criticality, even though it arrived later
    assert not q.outranked(crit)  # same level, later: FIFO, no jumping


def test_requeued_job_keeps_its_place_and_frees_the_slot():
    q = CallQueue(_FakeRedis())
    q.push(_job("high-1", "High", 1000))
    q.push(_job("high-2", "High", 2000))
    job = q.claim()
    q.push(_job("crit", "Critical", 3000))
    q.requeue(job)
    q.done(job.event_id)  # what the dispatcher does after handle() returns
    assert q.redis.get("rmsai:calls:active") is None
    assert q.claim().event_id == "crit"
    q.done("crit")
    assert q.claim().event_id == "high-1"  # ahead of high-2: original place kept


def test_retries_stop_for_a_higher_priority_call():
    import pytest  # noqa: PLC0415

    from common.config import DEFAULT  # noqa: PLC0415
    from voice.outbound import CallOutcome, CallPreempted, place_with_retries  # noqa: PLC0415

    caller = SimulatedCaller([CallOutcome.NO_ANSWER, CallOutcome.NO_ANSWER])
    with pytest.raises(CallPreempted):
        place_with_retries(caller, "+15551234567", DEFAULT, sleep_fn=lambda s: None,
                           preempt=lambda: True)
    assert caller.calls == 1  # no re-dial, no SMS fallback for the yielded call


def test_preempted_queued_call_goes_back_to_the_queue(patched, monkeypatch):  # noqa: F811
    from voice.outbound import CallPreempted  # noqa: PLC0415

    redis = _FakeRedis()
    q = CallQueue(redis)
    q.push(CallJob(event_id="ns", patient_ref="PT1", event_type="NORMAL_SINUS", bed="ICU/3",
                   criticality="High", reason="vitals_alert",
                   payload=event_to_dict(_event(_Afib())), enqueued_ms=1000))
    job = q.claim()
    q.push(_job("vt", "Critical", 2000))
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: "reported")

    def _place(*a, preempt, **k):
        assert preempt()  # the Critical is waiting
        raise CallPreempted

    monkeypatch.setattr(bus_consumer, "_place_live_call", _place)
    store = _FakeAlertStore()
    store.redis = redis
    assert bus_consumer.run_queued_call(
        job, driver=object(), orchestrator=object(), utterances=[], caller_factory=lambda r: object(),
        alert_store=store) is None
    q.done(job.event_id)
    assert q.claim().event_id == "vt"
    q.done("vt")
    assert q.claim().event_id == "ns"


def test_dropped_call_yields_without_a_stale_answer(patched, monkeypatch):  # noqa: F811
    redis = _FakeRedis()
    q = CallQueue(redis)
    q.push(CallJob(event_id="ns", patient_ref="PT1", event_type="NORMAL_SINUS", bed="ICU/3",
                   criticality="High", reason="vitals_alert",
                   payload=event_to_dict(_event(_Afib())), enqueued_ms=1000))
    job = q.claim()
    monkeypatch.setattr(bus_consumer, "get_event_status", lambda d, uuid: "reported")

    def _answered_then_dropped(*a, **k):  # PIN passed, then the line broke; a Critical arrives
        q.mark_answered("ns")
        q.mark_dropped("ns")
        q.push(_job("vt", "Critical", 2000))
        return bus_consumer.OutboundResult(called=True, decision_reason="ok", outcome="answered")

    monkeypatch.setattr(bus_consumer, "_place_live_call", _answered_then_dropped)
    store = _FakeAlertStore()
    store.redis = redis
    assert bus_consumer.run_queued_call(
        job, driver=object(), orchestrator=object(), utterances=[], caller_factory=lambda r: object(),
        alert_store=store) is None
    assert not q.reached("ns")  # the redial must earn its own answer
    q.done("ns")
    assert q.claim().event_id == "vt"
