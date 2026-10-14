"""P5: an unanswered alert call falls back to SMS; the SMS senders never crash an event."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from common.audit import AuditLog
from common.config import DEFAULT
from common.interfaces import ECGModel
from common.notify import SignalWireSmsNotifier, SimulatedSmsNotifier, notifier_from_env
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

    def run_write(self, _query, uuid, status, closed=()):
        if self.status.get(uuid) not in closed:  # set_event_status never reopens a closed event
            self.status[uuid] = status

    def run_read(self, _query, uuid):
        return [{"status": self.status.get(uuid)}]


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

class _RealSms(SimulatedSmsNotifier):
    """Stands in for a real sender: records the message, and its "delivered" counts."""

    simulated = False


def test_unanswered_call_texts_the_alert_and_stays_reported(tmp_path):
    sms = _RealSms()
    result, status, audits = _run([CallOutcome.NO_ANSWER] * 3, sms, tmp_path)
    assert result.outcome == "no_answer" and result.attempts == 3  # retries happened first
    assert result.fallback == "sms_delivered"
    assert result.status == status == "reported"  # alerted by text; still awaiting an ack
    (to, text), = sms.sent
    assert to == "+15551234567" and text.startswith("Missed call from RMS relay.")
    assert "atrial fibrillation" in text.lower()
    assert any(a["action"] == "sms_fallback" and a["outcome"] == "sms_delivered" for a in audits)


def test_a_simulated_sms_is_never_recorded_as_delivered(tmp_path, capsys):
    # `make demo-phone-up` ran without --notifier: after three unanswered dials the event read
    # "sms_delivered" / reported in the log and the app though no text was sent (2026-10-13).
    result, status, audits = _run([CallOutcome.NO_ANSWER] * 3, SimulatedSmsNotifier(), tmp_path)
    assert result.fallback == "sms_simulated"
    assert result.status == status == "notify_failed"  # nobody was reached
    assert any(a["action"] == "sms_fallback" and a["outcome"] == "sms_simulated" for a in audits)
    out = capsys.readouterr().out
    assert "SIMULATED, not sent" in out and "+15551234567" not in out  # says so, number masked


def test_ack_in_app_while_ringing_is_kept_and_not_texted(tmp_path):
    # The clinician acknowledged in the app while the queued call rang out: the unanswered call
    # set notify_failed (reopening the event) and texted "Missed call..." for a handled alarm.
    driver, ev = _Driver(), _event()
    driver.status[ev.window.event_id] = "acknowledged"
    sms = _RealSms()
    result = run_outbound(
        ev, driver=driver, orchestrator=None, caller=SimulatedCaller([CallOutcome.NO_ANSWER] * 3),
        utterances=[], config=_CFG, bed="Unit1-Bed05", audit=AuditLog(tmp_path / "a.jsonl"),
        sleep_fn=lambda _s: None, fallback_notifier=sms,
    )
    assert driver.status[ev.window.event_id] == "acknowledged"
    assert sms.sent == [] and result.fallback is None


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


# -------------------------------------------------------------- SignalWire sender (offline) ---

_SW_CFG = replace(_CFG, signalwire_space="example.signalwire.com", signalwire_project_id="proj-1",
                  signalwire_api_token="PTtok", outbound_from="+12015550100")


def test_signalwire_posts_to_the_space_laml_messages_api(capsys):
    calls = []

    def post(url, form, headers):
        calls.append((url, form, headers))
        return 201, '{"sid": "SMsw1"}'

    n = notifier_from_env("signalwire", config=_SW_CFG)
    assert isinstance(n, SignalWireSmsNotifier) and not n.simulated
    n._post = post
    assert n.send("+15551234567", "Missed call from RMS relay. Alert for PT1.")
    (url, form, headers), = calls
    assert url == ("https://example.signalwire.com/api/laml/2010-04-01/Accounts/proj-1/"
                   "Messages.json")
    assert form == {"To": "+15551234567", "From": "+12015550100",
                    "Body": "Missed call from RMS relay. Alert for PT1."}
    assert headers["Authorization"].startswith("Basic ")
    out = capsys.readouterr().out
    assert "sent via signalwire" in out and "SMsw1" in out and "+15551234567" not in out


def test_signalwire_space_may_be_a_bare_name():
    n = SignalWireSmsNotifier("example", "proj-1", "tok", "+12015550100", post_fn=lambda *a: (201, "{}"))
    assert n._url.startswith("https://example.signalwire.com/api/laml/")


def test_signalwire_rejection_returns_false(capsys):
    body = '{"code": 21211, "message": "Invalid To phone number"}'
    n = SignalWireSmsNotifier("example.signalwire.com", "proj-1", "tok", "+12015550100",
                              post_fn=lambda *a: (400, body))
    assert n.send("+15551234567", "hello") is False
    assert "21211" in capsys.readouterr().err


def test_notifier_from_env_names_missing_signalwire_keys():
    with pytest.raises(ValueError, match="SIGNALWIRE_API_TOKEN"):
        notifier_from_env("signalwire", config=replace(_SW_CFG, signalwire_api_token=""))


def _sw(status, body, calls):
    def post(url, form, headers):
        calls.append((url, form, headers))
        return status, body
    return SignalWireSmsNotifier("example.signalwire.com", "proj-1", "tok", "+12015550100",
                                 post_fn=post)


def test_signalwire_rejection_masks_the_number(capsys):
    assert not _sw(400, '{"code": 21211, "message": "Invalid To"}', []).send("+15551234567", "hi")
    err = capsys.readouterr().err
    assert "21211" in err and "+15551234567" not in err and "4567" in err


def test_signalwire_network_error_returns_false():
    def boom(*_a):
        raise OSError("network unreachable")
    n = SignalWireSmsNotifier("example", "proj-1", "tok", "+12015550100", post_fn=boom)
    assert n.send("+15551234567", "hello") is False


def test_signalwire_skips_invalid_destination():
    calls = []
    assert not _sw(201, "{}", calls).send("bogus", "hello") and calls == []


def test_the_sender_comes_from_the_hospital_config(monkeypatch):
    # The number lives in config/hospitals/<id>.yaml; an env-only lookup refused to start.
    monkeypatch.delenv("OUTBOUND_FROM", raising=False)
    assert notifier_from_env("signalwire", config=_SW_CFG)._from == "+12015550100"
    with pytest.raises(ValueError, match=r"outbound\.from"):
        notifier_from_env("signalwire", config=replace(_SW_CFG, outbound_from=""))
    assert isinstance(notifier_from_env("simulated"), SimulatedSmsNotifier)
    assert notifier_from_env("simulated", deliver=False).deliver is False
