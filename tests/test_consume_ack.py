"""cli.consume acks a message only once it is finished with: a failure is retried, then dead-lettered."""

from __future__ import annotations

import json

from cli.consume import handle_entry


class _FakeRedis:
    def __init__(self):
        self.acked, self.dead, self.hash = [], [], {}

    def xack(self, stream, group, msg_id):
        self.acked.append(msg_id)

    def hincrby(self, key, field, n):
        self.hash[field] = self.hash.get(field, 0) + n
        return self.hash[field]

    def hdel(self, key, field):
        self.hash.pop(field, None)

    def xadd(self, stream, fields, **kw):
        self.dead.append((stream, fields))


class _Audit:
    def __init__(self):
        self.outcomes = []

    def write(self, **kw):
        self.outcomes.append(kw["outcome"])


_FIELDS = {b"data": json.dumps({"uuid": "E1"}).encode()}


def _run(client, work, audit):
    return handle_entry(client, "s", "g", b"1-0", _FIELDS, work, audit=audit, max_attempts=3)


def test_success_is_acked():
    client, audit = _FakeRedis(), _Audit()
    assert _run(client, lambda p, m: None, audit) == "processed"
    assert client.acked == [b"1-0"]


def test_transient_failure_stays_pending_then_succeeds():
    client, audit = _FakeRedis(), _Audit()

    def down(p, m):
        raise ConnectionError("neo4j restarting")

    assert _run(client, down, audit) == "retry"
    assert client.acked == []  # left pending: the alarm is not lost
    assert _run(client, lambda p, m: None, audit) == "processed"
    assert client.acked == [b"1-0"] and client.hash == {}


def test_repeated_failure_is_dead_lettered_and_acked():
    client, audit = _FakeRedis(), _Audit()

    def broken(p, m):
        raise ValueError("bad event")

    outcomes = [_run(client, broken, audit) for _ in range(3)]
    assert outcomes == ["retry", "retry", "dead"]
    assert client.acked == [b"1-0"]
    assert client.dead[0][0] == "s.dead" and client.dead[0][1]["data"] == _FIELDS[b"data"]


def test_undecodable_message_is_poison_and_acked():
    client, audit = _FakeRedis(), _Audit()
    out = handle_entry(client, "s", "g", b"2-0", {b"data": b"{not json"}, lambda p, m: None,
                       audit=audit)
    assert out == "poison" and client.acked == [b"2-0"]
