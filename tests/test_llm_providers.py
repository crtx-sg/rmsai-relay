"""Cloud LLM providers (`common/providers.py`): routing, request shape, refusals, streaming, de-id.

No network: each provider takes an injectable SDK client, replaced here by a fake that records the
request and returns canned responses.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from common.config import DEFAULT
from common.deid import get_deidentifier
from common.providers import (
    AnthropicProvider,
    DeidentifyingLLM,
    EchoLLM,
    OpenAICompatProvider,
    OllamaProvider,
    get_llm_provider,
)

# --- fakes ---------------------------------------------------------------------------------------


class _FakeAnthropic:
    """`client.beta.messages.{create,stream}` with a scripted reply."""

    def __init__(self, text="rate control first", stop_reason="end_turn", chunks=None):
        self.calls = []
        self._text, self._stop, self._chunks = text, stop_reason, chunks
        self.beta = NS(messages=NS(create=self._create, stream=self._stream))

    def _create(self, **params):
        self.calls.append(params)
        content = [NS(type="thinking", thinking=""), NS(type="text", text=self._text)] if self._text else []
        return NS(stop_reason=self._stop, content=content)

    @contextmanager
    def _stream(self, **params):
        self.calls.append(params)
        chunks = self._chunks if self._chunks is not None else [self._text]
        yield NS(text_stream=iter(chunks), get_final_message=lambda: NS(stop_reason=self._stop))


class _FakeOpenAI:
    def __init__(self, text="rate control first"):
        self.calls = []
        self._text = text
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        if params.get("stream"):
            return iter([NS(choices=[NS(delta=NS(content=w))]) for w in ("rate ", "control")]
                        + [NS(choices=[])])  # usage-only trailer chunk
        return NS(choices=[NS(message=NS(content=self._text))])


# --- Anthropic -----------------------------------------------------------------------------------


def test_anthropic_request_shape_with_server_side_fallback():
    fake = _FakeAnthropic()
    out = AnthropicProvider(model="claude-opus-5-5", effort="low", client=fake).generate("Q?")
    p = fake.calls[0]
    assert out == "rate control first"  # text blocks only; the thinking block is skipped
    assert p["model"] == "claude-opus-5-5" and p["output_config"] == {"effort": "low"}
    assert p["messages"] == [{"role": "user", "content": "Q?"}] and p["system"]
    assert p["fallbacks"] == "default" and p["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_no_fallback_or_effort_on_models_without_them():
    fake = _FakeAnthropic()
    AnthropicProvider(model="claude-haiku-4-5", effort="low", client=fake).generate("Q?")
    p = fake.calls[0]
    assert "fallbacks" not in p and "betas" not in p and "output_config" not in p
    assert p["max_tokens"] == 16000  # the API requires a cap; LLM_MAX_TOKENS=0 means "default"


def test_anthropic_empty_effort_sends_none():
    fake = _FakeAnthropic()
    AnthropicProvider(model="claude-opus-5-5", effort="none", client=fake).generate("Q?")
    assert "output_config" not in fake.calls[0]


def test_anthropic_refusal_returns_a_fixed_sentence_not_partial_text():
    fake = _FakeAnthropic(text="partial", stop_reason="refusal")
    assert AnthropicProvider(client=fake).generate("Q?") == "The model declined to answer that."


def test_anthropic_streams_text_and_handles_a_refusal_before_output():
    p = AnthropicProvider(client=_FakeAnthropic(chunks=["rate ", "control"]))
    assert "".join(p.generate_stream("Q?")) == "rate control"
    refused = AnthropicProvider(client=_FakeAnthropic(chunks=[], stop_reason="refusal"))
    assert list(refused.generate_stream("Q?")) == ["The model declined to answer that."]


def test_anthropic_refusal_mid_stream_is_marked_not_silently_truncated():
    p = AnthropicProvider(client=_FakeAnthropic(chunks=["Rate con"], stop_reason="refusal"))
    assert "".join(p.generate_stream("Q?")) == "Rate con … The model declined to answer that."


# --- OpenAI-compatible (Gemini, OpenAI, others) --------------------------------------------------


def test_openai_compat_generate_and_stream():
    fake = _FakeOpenAI()
    p = OpenAICompatProvider(model="m", base_url="http://x", api_key="k", client=fake)
    assert p.generate("Q?") == "rate control first"
    assert fake.calls[0]["messages"][-1] == {"role": "user", "content": "Q?"}
    assert "".join(p.generate_stream("Q?")) == "rate control"
    assert fake.calls[1]["stream"] is True
    # no cap unless configured: reasoning models reject max_tokens, small hosted models cap lower
    assert "max_tokens" not in fake.calls[0] and "max_tokens" not in fake.calls[1]


def test_openai_compat_sends_the_cap_only_when_configured():
    fake = _FakeOpenAI()
    OpenAICompatProvider(model="m", base_url="http://x", api_key="k", max_tokens=800,
                         client=fake).generate("Q?")
    assert fake.calls[0]["max_tokens"] == 800


# --- routing from config -------------------------------------------------------------------------


def test_routing_by_llm_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    cfg = replace(DEFAULT, anthropic_model="claude-opus-5-5", anthropic_effort="medium",
                  gemini_model="gemini-x", openai_model="gpt-x",
                  openai_base_url="http://localhost:8000/v1")
    assert isinstance(get_llm_provider("echo", cfg), EchoLLM)
    assert isinstance(get_llm_provider("ollama", cfg), OllamaProvider)
    a = get_llm_provider("anthropic", cfg)
    assert isinstance(a, AnthropicProvider) and (a.model, a.effort) == ("claude-opus-5-5", "medium")
    g = get_llm_provider("gemini", cfg)
    assert isinstance(g, OpenAICompatProvider) and g.model == "gemini-x"
    assert "generativelanguage.googleapis.com" in str(g.client.base_url)
    assert g.client.api_key == "g-key"
    o = get_llm_provider("openai", cfg)
    assert o.model == "gpt-x" and str(o.client.base_url).startswith("http://localhost:8000/v1")


def test_unknown_provider_is_an_error_not_a_silent_stub():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_llm_provider("gpt", DEFAULT)


def test_cloud_provider_prints_a_synthetic_data_warning(monkeypatch, capsys):
    import common.providers as prov

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(prov, "_warned", set())
    get_llm_provider("gemini", DEFAULT)
    assert "SYNTHETIC data only" in capsys.readouterr().err


def test_prompts_are_deidentified_before_reaching_the_cloud():
    fake = _FakeAnthropic()
    llm = DeidentifyingLLM(AnthropicProvider(client=fake), get_deidentifier("regex"))
    llm.generate("Call 555-123-4567 about MRN 12345678")
    sent = fake.calls[0]["messages"][0]["content"]
    assert "555-123-4567" not in sent and "12345678" not in sent
