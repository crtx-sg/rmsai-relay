"""Outbound text notification — an alternative alert channel to the voice call (POC option).

A `Notifier` sends a short text message (SMS) to the configured destination. `SimulatedSmsNotifier`
records messages and can simulate a delivery failure (for the notify-failed path);
`TwilioSmsNotifier` is the real backend: Twilio's REST API over stdlib HTTP, no SDK dependency
(same approach as the ElevenLabs adapters). Same destination-number validation as the voice
caller, shared here so both channels agree on what a valid number is.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable

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


class TwilioSmsNotifier(Notifier):
    """Real SMS via Twilio's Messages REST API (stdlib HTTP).

    Never raises for a delivery problem: an unverified destination on a trial account, a bad
    from-number or a network error returns False with the reason printed (destination masked), so
    the caller takes its notify-failed path instead of crashing the event. `post_fn` is injectable
    for offline tests: `(url, form, headers) -> (status, body)`.
    """

    API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    def __init__(self, account_sid: str, auth_token: str, from_number: str,
                 post_fn: Callable[[str, dict, dict], tuple[int, str]] | None = None) -> None:
        self._sid = account_sid
        self._from = from_number
        token = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}",
                         "Content-Type": "application/x-www-form-urlencoded"}
        self._post = post_fn or _http_post

    def send(self, to: str, message: str) -> bool:
        if not is_valid_number(to):
            return False
        url = self.API.format(sid=self._sid)
        form = {"To": to, "From": self._from, "Body": message}
        try:
            status, body = self._post(url, form, self._headers)
        except OSError as exc:  # network down, DNS, timeout
            print(f"[sms] send to {_mask(to)} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return False
        if 200 <= status < 300:
            return True
        detail = body
        try:
            err = json.loads(body)
            detail = f"code {err.get('code')}: {err.get('message')}"
        except (ValueError, AttributeError):
            pass
        print(f"[sms] send to {_mask(to)} rejected ({status}) {detail}", file=sys.stderr)
        return False


def _http_post(url: str, form: dict, headers: dict) -> tuple[int, str]:  # pragma: no cover - network
    data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:  # 4xx/5xx carry Twilio's JSON error body
        return exc.code, exc.read().decode()


def _mask(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    return f"{number[:5]}…{digits[-4:]}" if len(digits) >= 8 else "…"


def notifier_from_env(name: str = "simulated", *, deliver: bool = True) -> Notifier:
    """`simulated`, or `twilio` built from TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / OUTBOUND_FROM.

    Raises ValueError naming whatever is missing, instead of a bare KeyError mid-startup.
    """
    if name != "twilio":
        return SimulatedSmsNotifier(deliver=deliver)
    env = {k: os.environ.get(k, "") for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                                              "OUTBOUND_FROM")}
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise ValueError(f"--notifier twilio needs {', '.join(missing)} in .env")
    return TwilioSmsNotifier(env["TWILIO_ACCOUNT_SID"], env["TWILIO_AUTH_TOKEN"],
                             env["OUTBOUND_FROM"])


def get_notifier(name: str = "simulated", **kwargs) -> Notifier:
    if name == "twilio":
        return TwilioSmsNotifier(**kwargs)
    return SimulatedSmsNotifier(**kwargs)
