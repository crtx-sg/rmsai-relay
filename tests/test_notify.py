"""Text notification (SMS) channel."""

from __future__ import annotations

from common.notify import SimulatedSmsNotifier, get_notifier, is_valid_number


def test_valid_number_shared():
    assert is_valid_number("+15551234567")
    assert not is_valid_number("nope")


def test_sms_send_records_message():
    n = SimulatedSmsNotifier()
    assert n.send("+15551234567", "alert: VT on bed 3")
    assert n.sent == [("+15551234567", "alert: VT on bed 3")]


def test_sms_delivery_failure():
    n = SimulatedSmsNotifier(deliver=False)
    assert not n.send("+15551234567", "alert")
    assert n.sent == []


def test_sms_invalid_number_fails():
    n = SimulatedSmsNotifier()
    assert not n.send("bogus", "alert")


def test_get_notifier_default_simulated():
    assert isinstance(get_notifier("simulated"), SimulatedSmsNotifier)
