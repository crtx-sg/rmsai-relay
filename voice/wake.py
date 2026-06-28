"""Wake-word detection for the audio Q&A phase.

After the outbound alert is delivered, follow-up *audio* turns must start with a wake word
("hey vios") before the agent answers — so room noise and Whisper hallucinations on silence don't
trigger spurious replies. Text-channel turns are never gated (they go through a separate handler).

`detect_wake_word` is a pure function (no model, no I/O) so it is unit-tested offline. It tolerates
the common Whisper mishearings of the brand word "vios" and strips the wake phrase so the handler
sees only the actual question.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
# Homophones Whisper tends to emit for "vios" (heard the brand word as these in live calls).
_VIOS_VARIANTS = "vios|bios|vias|veos|vyos|vials|viose|vio"


def _normalize(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def detect_wake_word(text: str, wake_word: str = "hey vios") -> tuple[bool, str]:
    """Return `(matched, remainder)` for an utterance.

    `remainder` is the utterance with everything up to and including the wake phrase removed (the
    actual question). Matches the wake phrase anywhere in the normalized text; for the default
    "hey vios" it also accepts common STT variants of "vios" and of the greeting ("hi/hello vios").
    """
    norm = _normalize(text)
    ww = _normalize(wake_word)
    if ww == "hey vios":
        pattern = re.compile(rf"\bhe(?:y|llo|i)\s+(?:{_VIOS_VARIANTS})\b")
    else:
        pattern = re.compile(rf"\b{re.escape(ww)}\b")
    m = pattern.search(norm)
    if not m:
        return False, ""
    return True, norm[m.end():].strip()
