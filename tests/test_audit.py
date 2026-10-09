"""Append-only JSONL audit log (G14)."""

from __future__ import annotations

import json

from common.audit import AuditLog


def test_write_and_read(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.write(actor="clinician", action="phi_read", subject="PT1234", outcome="ok")
    log.write(actor="system", action="outbound_call", subject="PT1234", outcome="acknowledged",
              number="+1555")
    records = log.read_all()
    assert len(records) == 2
    assert records[0]["subject"] == "PT1234"
    assert records[1]["extra"]["number"] == "+1555"
    assert "ts" in records[0]


def test_lines_are_valid_json(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.write(actor="a", action="query", subject="PT9", outcome="ok")
    for line in path.read_text().splitlines():
        json.loads(line)  # raises on malformed


def test_appends_not_overwrites(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).write(actor="a", action="x", subject="PT1", outcome="ok")
    AuditLog(path).write(actor="b", action="y", subject="PT2", outcome="ok")
    assert len(AuditLog(path).read_all()) == 2


def test_bare_auditlog_follows_audit_log_path(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(target))
    AuditLog().write(actor="a", action="x", subject="PT1", outcome="ok")
    assert target.exists() and len(AuditLog(target).read_all()) == 1


def test_bare_auditlog_defaults_without_env(monkeypatch):
    from common.audit import DEFAULT_AUDIT_LOG_PATH

    monkeypatch.delenv("AUDIT_LOG_PATH", raising=False)
    assert str(AuditLog().path) == DEFAULT_AUDIT_LOG_PATH
