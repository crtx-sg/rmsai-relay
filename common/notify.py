"""Outbound text notification — an alternative alert channel to the voice call (POC option).

A `Notifier` sends a short text message (SMS) to the configured destination. `SimulatedSmsNotifier`
records messages and can simulate a delivery failure (for the notify-failed path);
`TwilioSmsNotifier` is the real backend (lazy). Same destination-number validation as the voice
caller, shared here so both channels agree on what a valid number is.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_E164 = re.compile(r"^\+?\d{7,15}$")


def is_valid_number(number: str) -> bool:
    return bool(number) and bool(_E164.match(number.replace(" ", "").replace("-", "")))


class Notifier(ABC):
    @abstractmethod
    def send(self, to: str, message: str) -> bool:
        """Send a text message. Returns True on delivery, False on failure."""


class SimulatedSmsNotifier(Notifier):
    """Records sent messages; `deliver=False` simulates a delivery failure (notify-failed path)."""

    def __init__(self, deliver: bool = True) -> None:
        self.deliver = deliver
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, message: str) -> bool:
        if not is_valid_number(to) or not self.deliver:
            return False
        self.sent.append((to, message))
        return True


class TwilioSmsNotifier(Notifier):  # pragma: no cover - needs network + creds
    """Real SMS via Twilio (lazy import)."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        from twilio.rest import Client  # noqa: PLC0415

        self._client = Client(account_sid, auth_token)
        self._from = from_number

    def send(self, to: str, message: str) -> bool:
        if not is_valid_number(to):
            return False
        msg = self._client.messages.create(to=to, from_=self._from, body=message)
        return msg.sid is not None


def get_notifier(name: str = "simulated", **kwargs) -> Notifier:
    if name == "twilio":
        return TwilioSmsNotifier(**kwargs)
    return SimulatedSmsNotifier(**kwargs)
