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
# Homophones Whisper tends to emit for "vios" when it hears "hey vios" as two words.
_VIOS_VARIANTS = "vios|bios|vias|veos|vyos|vials|viose|vio|vos|veose"
# STT frequently merges "hey vios" into a single token (or mishears the lead-in) — these are the
# whole-phrase forms seen in live calls (e.g. "aveos", "avios", "a vios").
_COMPOUND_VARIANTS = r"aveos|avios|aveus|ahveos|aviose|ave\s+os|a\s+v[ie]os|hey\s+vos"


def _normalize(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def detect_wake_word(text: str, wake_word: str = "hey vios") -> tuple[bool, str]:
    """Return `(matched, remainder)` for an utterance.

    `remainder` is the utterance with everything up to and including the wake phrase removed (the
    actual question). Matches the wake phrase anywhere in the normalized text; for the default
    "hey vios" it also accepts common STT variants — both the two-word form ("hi/hello" + a "vios"
    homophone) and the merged single-token form ("aveos"/"avios"/"a vios"/...).
    """
    norm = _normalize(text)
    ww = _normalize(wake_word)
    if ww == "hey vios":
        pattern = re.compile(
            rf"\b(?:he(?:y|llo|i)\s+(?:{_VIOS_VARIANTS})|{_COMPOUND_VARIANTS})\b"
        )
    else:
        pattern = re.compile(rf"\b{re.escape(ww)}\b")
    m = pattern.search(norm)
    if not m:
        return False, ""
    return True, norm[m.end():].strip()
