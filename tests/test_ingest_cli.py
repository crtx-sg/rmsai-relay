"""Phase 1 ingest CLI: stdout summary + bus emit (Redis Streams)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from cli.ingest import main
from common.config import DEFAULT

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


def test_stdout_emits_one_summary_per_event(capsys):
    rc = main(["--file", str(_FIXTURE), "--emit", "stdout"])
    assert rc == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2  # fixture has 2 events
    for ln in lines:
        assert ln["event_type"]
        assert "criticality" in ln and "mews" in ln


def test_stdout_show_report(capsys):
    main(["--file", str(_FIXTURE), "--emit", "stdout", "--show-report"])
    out = capsys.readouterr().out
    assert "# Event Report" in out


@pytest.mark.infra
def test_bus_emit_publishes_to_redis_stream(capsys):
    redis = pytest.importorskip("redis")
    stream = f"rmsai.test.{uuid.uuid4().hex[:8]}"
    try:
        client = redis.Redis.from_url(DEFAULT.redis_url, socket_connect_timeout=2)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"redis unreachable: {exc}")

    rc = main(["--file", str(_FIXTURE), "--emit", "bus", "--stream", stream])
    assert rc == 0
    entries = client.xrange(stream)
    assert len(entries) == 2
    _, fields = entries[0]
    payload = json.loads(fields[b"data"])
    assert payload["event_type"]
    assert "report_md" in payload and "signals" not in payload  # raw signals excluded from bus
    assert payload["sample_rates"]["resp"] == "100/3"
    client.delete(stream)


def test_each_line_carries_an_outcome_and_metrics_prints_the_breakdown(capsys):
    rc = main(["--file", str(_FIXTURE), "--emit", "stdout", "--checkpoint", "--metrics"])
    assert rc == 0
    out = capsys.readouterr()
    lines = [json.loads(line) for line in out.out.splitlines() if line.strip()]
    assert all(ln["outcome"] in {"TP", "TP_WRONG_CLASS", "FP", "FN", "TN", "UNSCORABLE"}
               for ln in lines)
    summary = next(json.loads(ln) for ln in out.err.splitlines()
                   if ln.startswith('{"summary"'))["summary"]
    assert summary["tp"] + summary["fp"] + summary["fn"] + summary["tn"] == summary["scored"]
    assert "Model performance" in out.err and "sensitivity" in out.err


def test_explain_adds_a_why_headline(capsys):
    main(["--file", str(_FIXTURE), "--emit", "stdout", "--checkpoint", "--explain"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines and all(ln["why"].startswith(("Shown because", "Not alerted")) for ln in lines)
