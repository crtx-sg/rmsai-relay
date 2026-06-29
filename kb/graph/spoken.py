"""Normalize spoken (STT) queries so operational intents resolve the same as typed ones.

Whisper transcribes a clinician's speech, so structured identifiers arrive in spoken form that the
intent regexes (built for typed input) miss:

* spelled-out acronyms — "A V Block" for "AV Block", "S V T" for "SVT", "E C G" for "ECG";
* number words — "unit one bed oh one" for "Unit1-Bed01", "thirty minutes", "twenty four hours".

`normalize_spoken_query` collapses single-letter runs and rebuilds bed labels / numbers so
`match_intent` sees canonical text. It is a near-no-op for already-typed queries: digit labels and
multi-letter words are left untouched.
"""

from __future__ import annotations

import re

# A standalone single-letter token (optionally with trailing punctuation): "V", "T?", "S.".
_LETTER_TOKEN = re.compile(r"^([A-Za-z])([.,!?;:]*)$")

_ONES = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}

# A single spoken number word (for bed/unit labels, read as a digit sequence).
_NUM = r"(?:" + "|".join(sorted({*_ONES, *_TEENS}, key=len, reverse=True)) + r")"
# "unit one bed oh one" / "bed one" / "bed unit one bed oh one" -> a bed label. Number words after
# unit/bed become digits; an optional leading "bed" (from "...on bed unit one...") is absorbed so the
# rebuilt label isn't doubled.
_BED_SPOKEN = re.compile(
    rf"\b(?:bed\s+)?(?:unit\s+({_NUM})\s+)?bed\s+((?:{_NUM})(?:\s+{_NUM})*)\b", re.IGNORECASE)
# A spoken count before hours/minutes ("twenty four hours", "thirty minutes") -> additive digits.
_TIME_SPOKEN = re.compile(
    rf"\b((?:(?:{'|'.join(_TENS)})\s+)?(?:{_NUM})|(?:{'|'.join(_TENS)}))\s+(hours?|minutes?|mins?)\b",
    re.IGNORECASE,
)


def _collapse_acronyms(text: str) -> str:
    """Join runs of >=2 standalone single-letter tokens ("A V Block" -> "AV Block", "S V T" -> "SVT").

    Only whole single-letter tokens count, so a letter inside a hyphenated word ("v-tach") or a
    longer word is never folded — and a lone article ("a v-tach") stays put.
    """
    out: list[str] = []
    run: list[str] = []
    tail = ""  # trailing punctuation of the last letter in the run

    def flush() -> None:
        nonlocal run, tail
        if len(run) >= 2:
            out.append("".join(run).upper() + tail)
        elif run:
            out.append(run[0] + tail)
        run, tail = [], ""

    for tok in text.split():
        m = _LETTER_TOKEN.match(tok)
        if m:
            run.append(m.group(1))
            tail = m.group(2)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def _digits(phrase: str) -> str:
    """Number words read as a digit sequence: 'oh one' -> '01', 'one' -> '1', 'ten' -> '10'."""
    out = ""
    for w in phrase.lower().split():
        out += str(_ONES.get(w, _TEENS.get(w, "")))
    return out


def _additive(phrase: str) -> int | None:
    """Number words read additively: 'twenty four' -> 24, 'thirty' -> 30, 'ten' -> 10."""
    total = 0
    for w in phrase.lower().split():
        if w in _TENS:
            total += _TENS[w]
        elif w in _TEENS:
            total += _TEENS[w]
        elif w in _ONES:
            total += _ONES[w]
        else:
            return None
    return total


def _bed_sub(m: re.Match) -> str:
    unit = _digits(m.group(1)) if m.group(1) else "1"   # default to the single POC unit
    bed = _digits(m.group(2)).zfill(2)                  # bed labels are 2-digit ("Bed01")
    return f"bed Unit{unit}-Bed{bed}"


def _time_sub(m: re.Match) -> str:
    val = _additive(m.group(1))
    return f"{val} {m.group(2)}" if val is not None else m.group(0)


def normalize_spoken_query(query: str) -> str:
    """Collapse spelled acronyms + rebuild bed labels / numbers from a spoken (STT) query."""
    q = _collapse_acronyms(query)
    q = _BED_SPOKEN.sub(_bed_sub, q)
    q = _TIME_SPOKEN.sub(_time_sub, q)
    return q
