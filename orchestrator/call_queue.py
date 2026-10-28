"""Severity-ordered, Redis-backed queue of outbound phone calls, plus its single dispatcher.

The bus consumer used to place each call inline and wait for the clinician to hang up before it
read the next event, so the companion-app worklist only saw event N+1 after call N ended. Now the
consumer pushes every event to the app as soon as it is gated and *queues* the call here; one
`CallDispatcher` thread dials them one at a time.

Ordering: highest criticality first, first-come within a level. A Redis Stream can't do that
(it is strictly FIFO), so the queue is a sorted set of event ids (`rmsai:calls:queue`, lower score
pops first) next to a hash of job payloads (`rmsai:calls:jobs`). A job leaves the hash only once
its call has finished, so a job popped but unfinished when the process died is still in the hash
and `recover()` puts it back. The bus message itself is acked as soon as the event is persisted
and pushed to the app; the queue is what keeps the pending call.

One call at a time holds across processes too: a dispatcher must `claim()` the single call slot
(`rmsai:calls:active`, holding the event id being dialled, with a lease it keeps alive during the
call). Two consumers used to run two dispatchers, and a second consumer's startup `recover()`
re-queued the call the first one was placing, so the clinician was rung twice.

A call the clinician answered (right PIN, `mark_answered`) is not re-queued by `recover()`: a crash
after the PIN used to ring them a second time for the same event. Demo POC limits: a call cut off
between the PIN and the end of the spoken alert is not placed again (the event stays pending on the
worklist), and a restarted consumer can still re-queue a call that is ringing but not yet answered.

Jobs carry the bus payload (already on our own Redis, keyed by pseudonym), never names.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass

from common.criticality import criticality_rank

_QUEUE = "rmsai:calls:queue"
_JOBS = "rmsai:calls:jobs"
_ACTIVE = "rmsai:calls:active"  # the one call being placed system-wide (event id), with a lease
_ANSWERED = "rmsai:calls:answered:{event_id}"  # set by the voice worker on the right PIN
_ANSWERED_TTL_S = 24 * 3600
# Set by the voice worker when it hung up a call whose audio died on a LiveKit reconnect.
_DROPPED = "rmsai:calls:dropped:{event_id}"
_DROPPED_TTL_S = 3600
_LEASE_S = 60  # renewed every _LEASE_S/3 during a call; a crashed holder frees the slot after this
# Renew / release the slot only while it still holds OUR event id (atomic compare-and-act). A plain
# EXPIRE or GET-then-DEL could extend or free a slot another process had claimed after ours lapsed.
_RENEW = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
          "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end")
_RELEASE = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end")
# Score = (levels above this one) * _BAND + enqueue time in ms. Epoch ms (~1.8e12) stays well
# inside one band, so every Critical sorts before every High, FIFO within a level.
_BAND = 10**13
_LEVELS = 4  # Low, Medium, High, Critical


def priority_score(criticality: str, enqueued_ms: int) -> int:
    rank = max(criticality_rank(criticality), 0)
    return (_LEVELS - 1 - rank) * _BAND + enqueued_ms


@dataclass
class CallJob:
    """One pending outbound call: the bus payload to rebuild the event from, plus its routing."""

    event_id: str
    patient_ref: str
    event_type: str
    bed: str
    criticality: str
    reason: str  # the criticality-gate reason (e.g. "ok", "fp_override ...")
    payload: dict
    enqueued_ms: int = 0

    @property
    def score(self) -> int:
        return priority_score(self.criticality, self.enqueued_ms)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CallJob":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls(**json.loads(raw))


class CallQueue:
    """Push / pop / done over the sorted set + job hash. One dispatcher pops; any process pushes."""

    def __init__(self, redis_client, *, queue_key: str = _QUEUE, jobs_key: str = _JOBS,
                 active_key: str = _ACTIVE, lease_s: int = _LEASE_S) -> None:
        self.redis = redis_client
        self.queue_key = queue_key
        self.jobs_key = jobs_key
        self.active_key = active_key
        self.lease_s = lease_s

    def push(self, job: CallJob) -> int:
        """Queue `job` (idempotent per event id: a redelivered event keeps its place). Returns the
        number of calls now waiting."""
        if not job.enqueued_ms:
            job.enqueued_ms = int(time.time() * 1000)
        if self.redis.hsetnx(self.jobs_key, job.event_id, job.to_json()):
            self.redis.zadd(self.queue_key, {job.event_id: job.score})
        return self.pending()

    def pop(self) -> CallJob | None:
        """The highest-priority waiting job, or None. It stays in the job hash until `done()`."""
        while True:
            popped = self.redis.zpopmin(self.queue_key, 1)
            if not popped:
                return None
            event_id = popped[0][0]
            raw = self.redis.hget(self.jobs_key, event_id)
            if raw is not None:
                return CallJob.from_json(raw)
            # A queue entry without a job (cleared by hand): skip it.

    def claim(self) -> CallJob | None:
        """Take the system-wide call slot and pop the next job; None if another process holds the
        slot or nothing is waiting. While the slot is held, `keep_alive()` it; `done()` frees it.

        Holding the slot means no other process is mid-call, so any popped-but-unfinished job is
        an orphan of a crashed holder: it is re-queued (in severity order) before popping.
        """
        if not self.redis.set(self.active_key, "claiming", nx=True, ex=self.lease_s):
            return None
        self.recover()
        job = self.pop()
        if job is None:
            self.redis.delete(self.active_key)
            return None
        self.redis.set(self.active_key, job.event_id, ex=self.lease_s)
        return job

    def keep_alive(self, event_id: str) -> bool:
        """Renew the call-slot lease for `event_id`; False if the slot is no longer ours (it lapsed,
        e.g. the laptop slept through a call, and another process may now be dialling)."""
        return bool(self.redis.eval(_RENEW, 1, self.active_key, event_id, self.lease_s))

    def mark_answered(self, event_id: str) -> None:
        """The clinician passed the PIN on this event's call: `recover()` must not ring them again.
        A key of its own (one SET, not an edit of the job JSON) so it can't recreate a job `done()`
        already deleted."""
        self.redis.set(_ANSWERED.format(event_id=event_id), "1", ex=_ANSWERED_TTL_S)

    def mark_dropped(self, event_id: str) -> None:
        """The worker ended this event's call because the connection broke: ring it again."""
        self.redis.set(_DROPPED.format(event_id=event_id), "1", ex=_DROPPED_TTL_S)

    def pop_dropped(self, event_id: str) -> bool:
        """True (once) if the worker marked this event's call as dropped."""
        return bool(self.redis.delete(_DROPPED.format(event_id=event_id)))

    def done(self, event_id: str) -> None:
        self.redis.hdel(self.jobs_key, event_id)
        self.redis.eval(_RELEASE, 1, self.active_key, event_id)

    def pending(self) -> int:
        return int(self.redis.zcard(self.queue_key))

    def recover(self) -> int:
        """Re-queue jobs that were popped but never finished (the process died mid-call). The job
        another process is dialling right now (it holds the call slot) is left alone, and a job whose
        call was already answered is dropped instead.

        shortcut: "answered" = the right PIN, so a call cut off before the alert finished is not
        re-placed, and a call still ringing can be re-queued; track call state in the worker if a
        real deployment needs either."""
        queued = {_s(m) for m in self.redis.zrange(self.queue_key, 0, -1)}
        active = self.redis.get(self.active_key)
        if active is not None:
            queued.add(_s(active))
        n = 0
        for event_id, raw in self.redis.hgetall(self.jobs_key).items():
            event_id = _s(event_id)
            if event_id in queued:
                continue
            if self.redis.get(_ANSWERED.format(event_id=event_id)) is not None:
                self.redis.hdel(self.jobs_key, event_id)
                print(f"[calls] not re-calling {event_id}: already answered before the crash",
                      flush=True)
            else:
                self.redis.zadd(self.queue_key, {event_id: CallJob.from_json(raw).score})
                n += 1
        return n


def _s(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


class CallDispatcher:
    """Single background thread that places queued calls one after another.

    `handle(job)` places one call and returns once it is over (answered and hung up, or given up
    on). A failing job is logged and dropped rather than retried forever; a crash mid-call leaves
    the job in the hash for `CallQueue.recover()`.
    """

    def __init__(self, queue: CallQueue, handle, *, poll_s: float = 1.0) -> None:
        self.queue = queue
        self.handle = handle
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="call-dispatcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        """Wake the dispatcher now (a job was just queued) instead of at its next poll."""
        self._wake.set()

    def step(self) -> bool:
        """Place the next queued call, if any. Returns False when the queue was empty."""
        self._busy.set()
        try:
            job = self.queue.claim()
            if job is None:
                return False
            beating = threading.Event()

            def _heartbeat() -> None:  # keep the call slot while the call lasts
                while not beating.wait(self.queue.lease_s / 3):
                    try:
                        if not self.queue.keep_alive(job.event_id):
                            print(f"[calls] call slot lost during the call for {job.event_id}",
                                  flush=True)
                            return
                    except Exception as exc:  # noqa: BLE001 - a Redis blip; retry next beat
                        print(f"[calls] lease renewal failed: {type(exc).__name__}", flush=True)

            threading.Thread(target=_heartbeat, name="call-lease", daemon=True).start()
            try:
                self.handle(job)
            except Exception as exc:  # noqa: BLE001 - one bad call must not stop the others
                print(f"[calls] call for event {job.event_id} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            finally:
                try:
                    self.queue.done(job.event_id)
                except Exception as exc:  # noqa: BLE001 - the lease lapses; recover() re-queues it
                    print(f"[calls] could not finish {job.event_id}: {type(exc).__name__}",
                          flush=True)
                finally:
                    beating.set()
            return True
        finally:
            self._busy.clear()

    def idle(self) -> bool:
        return not self._busy.is_set() and self.queue.pending() == 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.step():
                    continue
            except Exception as exc:  # noqa: BLE001 - e.g. Redis blip: back off, keep the thread
                print(f"[calls] queue read failed: {type(exc).__name__}: {exc}", flush=True)
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def stop(self, *, drain: bool = False, timeout: float | None = None) -> None:
        """Stop the thread; with `drain`, first wait (up to `timeout`) for the queue to empty."""
        if drain:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self.idle() and (deadline is None or time.monotonic() < deadline):
                self.notify()
                time.sleep(0.2)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)
