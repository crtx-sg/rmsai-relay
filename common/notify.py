"""Outbound text notification — an alternative alert channel to the voice call (POC option).

A `Notifier` sends a short text message (SMS) to the configured destination. `SimulatedSmsNotifier`
records messages and can simulate a delivery failure (for the notify-failed path); it sends
nothing, says so in the log, and the fallback records it as `sms_simulated`, never as delivered.
The real backend is `SignalWireSmsNotifier`: SignalWire's REST Messages API from the number that
also places the calls, over stdlib HTTP, no SDK dependency (same approach as the ElevenLabs
adapters). Same destination-number validation as the voice
caller, shared here so both channels agree on what a valid number is.
"""

from __future__ import annotations

import base64
import json
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
    # True for a notifier that sends nothing: a "delivered" result from it is not a delivery.
    simulated: bool = False

    @abstractmethod
    def send(self, to: str, message: str) -> bool:
        """Send a text message. Returns True on delivery, False on failure."""


class SimulatedSmsNotifier(Notifier):
    """Records sent messages; `deliver=False` simulates a delivery failure (notify-failed path)."""

    simulated = True

    def __init__(self, deliver: bool = True) -> None:
        self.deliver = deliver
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, message: str) -> bool:
        if not is_valid_number(to) or not self.deliver:
            return False
        self.sent.append((to, message))
        print(f"[sms] SIMULATED, not sent: text to {_mask(to)} "
              "(pass --notifier signalwire to send it)", flush=True)
        return True


class SignalWireSmsNotifier(Notifier):
    """Real SMS via SignalWire's REST Messages API (stdlib HTTP).

    Project ID + API token as basic auth on the SignalWire space. The sender is the SignalWire
    number that also places the calls (`outbound.from`), so the clinician sees the missed call and
    the text from one number.

    Never raises for a delivery problem: a rejected destination, a bad from-number or a network
    error returns False with the reason printed (destination masked), so the caller takes its
    notify-failed path instead of crashing the event. `post_fn` is injectable for offline tests:
    `(url, form, headers) -> (status, body)`.
    """

    API = "https://{host}/api/laml/2010-04-01/Accounts/{project}/Messages.json"

    def __init__(self, space: str, project_id: str, api_token: str, from_number: str,
                 post_fn: Callable[[str, dict, dict], tuple[int, str]] | None = None) -> None:
        host = space if "." in space else f"{space}.signalwire.com"
        self._url = self.API.format(host=host, project=project_id)
        self._from = from_number
        token = base64.b64encode(f"{project_id}:{api_token}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}",
                         "Content-Type": "application/x-www-form-urlencoded"}
        self._post = post_fn or _http_post

    def send(self, to: str, message: str) -> bool:
        if not is_valid_number(to):
            return False
        form = {"To": to, "From": self._from, "Body": message}
        try:
            status, body = self._post(self._url, form, self._headers)
        except OSError as exc:  # network down, DNS, timeout
            print(f"[sms] send to {_mask(to)} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return False
        if 200 <= status < 300:
            sid = ""
            try:
                sid = json.loads(body).get("sid") or ""
            except (ValueError, AttributeError):
                pass
            print(f"[sms] sent via signalwire to {_mask(to)} (message {sid or '?'})", flush=True)
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
    except urllib.error.HTTPError as exc:  # 4xx/5xx carry the provider's JSON error body
        return exc.code, exc.read().decode()


def _mask(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    return f"{number[:5]}…{digits[-4:]}" if len(digits) >= 8 else "…"


NOTIFIERS = ("simulated", "signalwire")


def notifier_from_env(name: str = "simulated", *, deliver: bool = True,
                      config=None) -> Notifier:
    """`simulated` or `signalwire`, built from the config.

    The sender is the hospital's `outbound.from` (env `OUTBOUND_FROM` overrides it), read through
    `Config`, not the raw environment, since the per-hospital file holds the numbers. Raises
    ValueError naming whatever is missing, instead of a bare KeyError mid-startup.
    """
    if name != "signalwire":
        return SimulatedSmsNotifier(deliver=deliver)
    if config is None:
        from common.config import Config  # noqa: PLC0415

        config = Config.from_env()
    need = {"SIGNALWIRE_SPACE": config.signalwire_space,
            "SIGNALWIRE_PROJECT_ID": config.signalwire_project_id,
            "SIGNALWIRE_API_TOKEN": config.signalwire_api_token,
            "outbound.from (hospital file)": config.outbound_from}
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise ValueError(f"--notifier signalwire needs {', '.join(missing)}")
    return SignalWireSmsNotifier(config.signalwire_space, config.signalwire_project_id,
                                 config.signalwire_api_token, config.outbound_from)


def get_notifier(name: str = "simulated", **kwargs) -> Notifier:
    if name == "signalwire":
        return SignalWireSmsNotifier(**kwargs)
    return SimulatedSmsNotifier(**kwargs)
