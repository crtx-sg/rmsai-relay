"""Wake-word detection for the audio Q&A phase.

After the outbound alert, follow-up *audio* turns must open with the wake word ("hey vios") before
the agent answers — so room noise and Whisper hallucinations on silence don't trigger replies. Text
turns are never gated.

The hard part: "vios" is out-of-vocabulary for Whisper, so it substitutes similar-sounding real
words ("bios", "why us", "aveos", "a vios", ...) that change call to call. Rather than enumerate
every mishearing, detection is **greeting-anchored**: if the utterance opens with a greeting word
(which Whisper transcribes reliably), treat it as a wake and strip the mangled "vios" span (the
1–2 non-question tokens after the greeting) so the handler sees only the real question. A few merged
single-token forms ("aveos"/"a vios") are matched directly. Pure function — unit-tested offline.
(`STT_INITIAL_PROMPT` also primes Whisper with "Vios" so it mis-hears the word less often.)
"""

from __future__ import annotations

import re

_APOS = re.compile(r"['’]")          # drop apostrophes: "what's" -> "whats"
_PUNCT = re.compile(r"[^\w\s]")

# Greeting tokens that open the wake phrase. Whisper keeps these even when it mangles "vios". Kept
# tight (no "ok"/"okay"/"yeah") so silence-hallucination noise doesn't read as a wake.
_GREETINGS = {"hey", "hi", "hello", "yo", "hay", "heya"}

# Whole-phrase mishearings where "hey vios" merges into one/two tokens with no clean greeting word.
_MERGED_RE = re.compile(r"^(?:aveos|avios|aveus|ahveos|aviose|avos|a\s+v[ie]os|ave\s+os)\b")

# Words that mark the START of the real question — used to stop stripping the mangled "vios" span.
# Deliberately excludes "why" (a common "vios" mishearing). The strip is also hard-capped, so a
# missing starter only ever costs a token or two.
_STARTERS = {
    "what", "whats", "which", "who", "whose", "show", "get", "tell", "give", "list", "is", "are",
    "was", "were", "do", "does", "did", "can", "could", "would", "how", "when", "where", "status",
    "find", "read", "describe", "explain", "current", "latest", "any", "has", "have", "report",
    "reports", "vitals", "vital", "patient", "summarize", "summary", "check", "acknowledge", "ack",
}
_MAX_JUNK = 2  # at most this many mangled-"vios" tokens dropped between greeting and question


def _normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", _APOS.sub("", text.lower())).split())


def detect_wake_word(text: str, wake_word: str = "hey vios") -> tuple[bool, str]:
    """Return `(matched, remainder)` — remainder is the utterance with the wake phrase removed.

    For the default "hey vios" this is greeting-anchored + mangling-tolerant (see module docstring).
    A non-default `wake_word` is matched as a literal phrase anywhere in the normalized text.
    """
    norm = _normalize(text)
    if not norm:
        return False, ""

    if wake_word.strip().lower() != "hey vios":
        ww = _normalize(wake_word)
        i = norm.find(ww)
        return (True, norm[i + len(ww):].strip()) if i >= 0 else (False, "")

    # Merged single-token forms ("aveos", "a vios", ...) with no separate greeting word.
    m = _MERGED_RE.match(norm)
    if m:
        return True, norm[m.end():].strip()

    # Greeting-anchored: find a greeting in the opening tokens (tolerates a leading "um"/"uh"),
    # then drop the mangled "vios" span (leading non-starter tokens, hard-capped).
    toks = norm.split()
    for g in range(min(3, len(toks))):
        if toks[g] in _GREETINGS:
            rest = toks[g + 1:]
            dropped = 0
            while rest and dropped < _MAX_JUNK and rest[0] not in _STARTERS:
                rest.pop(0)
                dropped += 1
            return True, " ".join(rest)
    return False, ""
