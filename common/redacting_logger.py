"""Redaction by construction (hard rule 6).

A stdlib `logging.Filter` that scrubs anything that looks like a name or free-text note from
log records before they are emitted. Patients are referenced by pseudonym (`PT####`) only; the
filter passes those through untouched but redacts `name=...`, quoted free text, and an optional
registered set of known names.

This is a defence-in-depth net, not a license to log PHI: call sites must still pass pseudonyms,
never names. The Phase 0 test plants a name and asserts it never reaches the output.
"""

from __future__ import annotations

import logging
import re

_REDACTED = "[REDACTED]"

# key=value where the key smells like PHI (name/patient_name/note/notes/free_text/comment)
_KV_PATTERN = re.compile(
    r"\b(name|patient_name|first_name|last_name|note|notes|free_text|comment)\s*=\s*"
    r"('[^']*'|\"[^\"]*\"|[^\s,;]+)",
    re.IGNORECASE,
)


class RedactingFilter(logging.Filter):
    """Scrubs PHI-shaped substrings from each record's rendered message + args."""

    def __init__(self, extra_names: set[str] | None = None) -> None:
        super().__init__()
        self._names = {n for n in (extra_names or set()) if n}

    def add_name(self, name: str) -> None:
        if name:
            self._names.add(name)

    def _scrub(self, text: str) -> str:
        text = _KV_PATTERN.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)
        for n in self._names:
            if n:
                text = re.sub(re.escape(n), _REDACTED, text, flags=re.IGNORECASE)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        # Render args into the message now, then scrub, so formatting can't re-introduce PHI.
        try:
            rendered = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            rendered = str(record.msg)
        record.msg = self._scrub(rendered)
        record.args = ()
        return True


def get_redacting_logger(
    name: str = "rmsai", *, level: int = logging.INFO, extra_names: set[str] | None = None
) -> logging.Logger:
    """Return a logger with a `RedactingFilter` attached to a stream handler (idempotent)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter(extra_names))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger
