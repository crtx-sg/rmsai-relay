"""Caller authentication — shared-PIN gate (G10, POC).

Verifies a PIN spoken as words ("one two three four"), spoken as a number ("twelve thirty-four"
is NOT supported — digits only), or entered as DTMF/literal digits ("1234"), before any PHI is
voiced. Per-user identity / voice enrolment is deferred (O2).
"""

from __future__ import annotations

import re

from common.config import DEFAULT, Config

_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def parse_pin(text: str) -> str:
    """Extract a digit string from spoken or DTMF input ('one two 3 4' -> '1234')."""
    digits: list[str] = []
    for token in re.findall(r"[a-z]+|\d", text.lower()):
        if token.isdigit():
            digits.append(token)
        elif token in _WORD_TO_DIGIT:
            digits.append(_WORD_TO_DIGIT[token])
    return "".join(digits)


class PinAuthGate:
    def __init__(self, config: Config = DEFAULT) -> None:
        self._pin = config.inbound_auth_pin

    def verify(self, spoken_or_digits: str) -> bool:
        parsed = parse_pin(spoken_or_digits)
        return bool(parsed) and parsed == self._pin
