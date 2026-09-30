"""P5: an unanswered alert call falls back to SMS; the Twilio sender never crashes an event."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from common.audit import AuditLog
from common.config import DEFAULT
from common.interfaces import ECGModel
from common.notify import SimulatedSmsNotifier, TwilioSmsNotifier, notifier_from_env
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from orchestrator.outbound_flow import run_outbound
from voice.outbound import CallOutcome, SimulatedCaller

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))
_CFG = replace(DEFAULT, outbound_enabled=True, outbound_call_number="+15551234567",
               outbound_min_criticality="High", outbound_max_retries=2)


class _Afib(ECGModel):
    def predict(self, window):
        return "ATRIAL_FIBRILLATION", 0.92


class _Driver:
    """Records status writes; the flow only touches the graph through set_event_status here."""

    def __init__(self):
        self.status: dict[str, str] = {}

    def run_write(self, _query, uuid, status):
        self.status[uuid] = status


def _event():
    w = next(read_hdf5_file(_FIXTURE))
    w.vitals["HR"].value = 145.0
    return process_window(w, _Afib(), MewsVitalsAnalysis())


def _run(outcomes, notifier, tmp_path, config=_CFG):
    driver, ev = _Driver(), _event()
    audit_path = tmp_path / "audit.jsonl"
    result = run_outbound(
        ev, driver=driver, orchestrator=None, caller=SimulatedCaller(outcomes), utterances=[],
        config=config, bed="Unit1-Bed05", audit=AuditLog(audit_path), sleep_fn=lambda _s: None,
        fallback_notifier=notifier,
    )
    audits = [json.loads(ln) for ln in audit_path.read_text().splitlines()] if audit_path.exists() else []
    return result, driver.status[ev.window.event_id], audits


# ------------------------------------------------------------------------ run_outbound path ---

def test_unanswered_call_texts_the_alert_and_stays_reported(tmp_path):
    sms = SimulatedSmsNotifier()
    result, status, audits = _run([CallOutcome.NO_ANSWER] * 3, sms, tmp_path)
    assert result.outcome == "no_answer" and result.attempts == 3  # retries happened first
    assert result.fallback == "sms_delivered"
    assert result.status == status == "reported"  # alerted by text; still awaiting an ack
    (to, text), = sms.sent
    assert to == "+15551234567" and text.startswith("Missed call from RMS relay.")
    assert "atrial fibrillation" in text.lower()
    assert any(a["action"] == "sms_fallback" and a["outcome"] == "sms_delivered" for a in audits)


def test_fallback_delivery_failure_is_notify_failed(tmp_path):
    result, status, audits = _run([CallOutcome.NO_ANSWER] * 3,
                                  SimulatedSmsNotifier(deliver=False), tmp_path)
    assert result.fallback == "sms_failed" and result.status == status == "notify_failed"
    assert any(a["action"] == "sms_fallback" and a["outcome"] == "sms_failed" for a in audits)


def test_no_notifier_keeps_previous_behaviour(tmp_path):
    result, status, _ = _run([CallOutcome.NO_ANSWER] * 3, None, tmp_path)
    assert result.fallback is None and status == "notify_failed"


def test_answered_call_sends_no_sms(tmp_path, monkeypatch):
    # answered + live_audio=False runs the scripted converse; keep it out of this test
    import orchestrator.outbound_flow as flow

    monkeypatch.setattr(flow, "_converse", lambda *a, **k: {
        "answers": [], "acknowledged": True, "dropped": False, "status": "acknowledged",
        "transcript": ["alert"]})
    sms = SimulatedSmsNotifier()
    result, _, _ = _run([CallOutcome.ANSWERED], sms, tmp_path)
    assert result.fallback is None and sms.sent == []


def test_invalid_number_is_never_texted(tmp_path):
    sms = SimulatedSmsNotifier()
    result, status, _ = _run([CallOutcome.ANSWERED], sms, tmp_path,
                             config=replace(_CFG, outbound_call_number="bogus"))
    assert result.outcome == "invalid" and result.fallback is None and sms.sent == []
    assert status == "notify_failed"


# ------------------------------------------------------------------ Twilio sender (offline) ---

def _twilio(status, body, calls):
    def post(url, form, headers):
        calls.append((url, form, headers))
        return status, body
    return TwilioSmsNotifier("ACsid", "tok", "+15550001000", post_fn=post)


def test_twilio_posts_the_message_form():
    calls = []
    assert _twilio(201, '{"sid": "SM1"}', calls).send("+15551234567", "hello")
    (url, form, headers), = calls
    assert url == "https://api.twilio.com/2010-04-01/Accounts/ACsid/Messages.json"
    assert form == {"To": "+15551234567", "From": "+15550001000", "Body": "hello"}
    assert headers["Authorization"].startswith("Basic ")


def test_twilio_rejection_returns_false_with_masked_reason(capsys):
    body = '{"code": 21608, "message": "The number is unverified. Trial accounts cannot send..."}'
    assert not _twilio(400, body, []).send("+15551234567", "hello")
    err = capsys.readouterr().err
    assert "21608" in err and "unverified" in err
    assert "+15551234567" not in err and "4567" in err  # masked


def test_twilio_network_error_returns_false():
    def boom(*_a):
        raise OSError("network unreachable")
    n = TwilioSmsNotifier("ACsid", "tok", "+15550001000", post_fn=boom)
    assert n.send("+15551234567", "hello") is False


def test_twilio_skips_invalid_destination():
    calls = []
    assert not _twilio(201, "{}", calls).send("bogus", "hello") and calls == []


def test_notifier_from_env_names_missing_twilio_keys(monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "OUTBOUND_FROM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACsid")
    with pytest.raises(ValueError, match="TWILIO_AUTH_TOKEN, OUTBOUND_FROM"):
        notifier_from_env("twilio")
    assert isinstance(notifier_from_env("simulated"), SimulatedSmsNotifier)
    assert notifier_from_env("simulated", deliver=False).deliver is False
