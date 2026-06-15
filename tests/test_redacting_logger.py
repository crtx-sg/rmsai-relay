"""Redaction by construction (hard rule 6): no names reach the output."""

from __future__ import annotations

import io
import logging

from common.redacting_logger import RedactingFilter, get_redacting_logger


def _capture(logger: logging.Logger) -> io.StringIO:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return buf


def test_redacts_name_kv():
    logger = get_redacting_logger("rmsai.test.kv", extra_names=set())
    buf = _capture(logger)
    logger.info("event for patient_name='John Doe' pseudonym=PT1234")
    out = buf.getvalue()
    assert "John Doe" not in out
    assert "[REDACTED]" in out
    assert "PT1234" in out  # pseudonyms pass through


def test_redacts_registered_name():
    f = RedactingFilter(extra_names={"Alice Smith"})
    logger = logging.getLogger("rmsai.test.name")
    logger.handlers.clear()
    logger.filters.clear()
    logger.addFilter(f)
    logger.setLevel(logging.INFO)
    buf = _capture(logger)
    logger.info("note about Alice Smith here")
    out = buf.getvalue()
    assert "Alice Smith" not in out
    assert "[REDACTED]" in out


def test_args_are_rendered_then_scrubbed():
    logger = get_redacting_logger("rmsai.test.args", extra_names=set())
    buf = _capture(logger)
    logger.info("name=%s ok", "Bob")
    out = buf.getvalue()
    assert "Bob" not in out
