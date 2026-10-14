"""Env parsing: a blank value means "unset" (use the default), never a crash or a silent False."""

from __future__ import annotations

from common.config import _b, _f, _i


def test_blank_numbers_fall_back_to_default(monkeypatch):
    monkeypatch.setenv("RMSAI_T_INT", "")
    monkeypatch.setenv("RMSAI_T_FLOAT", "  ")
    assert _i("RMSAI_T_INT", 7) == 7
    assert _f("RMSAI_T_FLOAT", 0.5) == 0.5


def test_blank_bool_keeps_its_default(monkeypatch):
    monkeypatch.setenv("RMSAI_T_BOOL", "")
    assert _b("RMSAI_T_BOOL", True) is True
    monkeypatch.setenv("RMSAI_T_BOOL", "off")
    assert _b("RMSAI_T_BOOL", True) is False


def test_set_values_still_win(monkeypatch):
    monkeypatch.setenv("RMSAI_T_INT", "12")
    assert _i("RMSAI_T_INT", 7) == 12
    monkeypatch.delenv("RMSAI_T_INT")
    assert _i("RMSAI_T_INT", 7) == 7
