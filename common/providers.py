"""LLM providers + a de-identifying wrapper.

* `EchoLLM` — deterministic test provider (captures prompts, echoes the question). No network.
* `OllamaProvider` — local self-hosted LLM (POC default for real use), stdlib HTTP, lazy.
* `AnthropicProvider` — Claude via the official `anthropic` SDK (`--extra llm-cloud`).
* `OpenAICompatProvider` — any OpenAI-compatible chat API via the `openai` SDK: Gemini (its
  OpenAI-compatible endpoint), OpenAI, Mistral, Groq, OpenRouter, vLLM, LM Studio, ...
  Cloud providers are for SYNTHETIC data only (hard rules 4/5); `build_llm` and every caller wrap
  them in `DeidentifyingLLM`, and selecting one prints a warning.
* `DeidentifyingLLM` — wraps any provider so **every** `generate`/`embed` call de-identifies its
  input first (fail closed). This is how "PHI is scrubbed before any model call" is enforced
  centrally: the orchestrator only ever holds a de-identifying provider.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from .config import DEFAULT, Config
from .deid import Deidentifier, deidentify, get_deidentifier
from .interfaces import LLMProvider


class EchoLLM(LLMProvider):
    """Records every prompt it receives and returns a deterministic echo of the question."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        question = ""
        for line in prompt.splitlines():
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
        return f"[grounded answer to: {question or prompt.splitlines()[-1][:80]}]"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from kb.vector.embeddings import HashingEmbedder  # noqa: PLC0415

        return HashingEmbedder().embed(texts)

    @property
    def last_prompt(self) -> str | None:
        return self.prompts[-1] if self.prompts else None


class OllamaProvider(LLMProvider):
    """Self-hosted LLM via Ollama's HTTP API (stdlib only)."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or DEFAULT.ollama_url).rstrip("/")
        self.model = model or DEFAULT.llm_model

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted local host)
            return json.loads(resp.read())

    def generate(self, prompt: str, **kwargs) -> str:
        return self._post("/api/generate", {"model": self.model, "prompt": prompt, "stream": False})["response"]

    def generate_stream(self, prompt: str, **kwargs):
        """Stream tokens from Ollama (stream=True) so TTS can start on the first sentence."""
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps({"model": self.model, "prompt": prompt, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted local host)
            for line in resp:
                if line.strip():
                    tok = json.loads(line).get("response", "")
                    if tok:
                        yield tok

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._post("/api/embeddings", {"model": self.model, "prompt": t})["embedding"] for t in texts]


# A short system instruction for every cloud call: answers are read on a phone or spoken aloud.
_LATENCY_SYSTEM = "Latency-sensitive; begin your visible answer immediately."
# Claude models that accept server-side refusal fallbacks in the `"default"` form.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5",
                    "claude-sonnet-5-5")
# Claude models that reject `output_config.effort` (Haiku, and the generations before it existed;
# `-4-2…` catches the date-suffixed Claude 4.0 ids such as claude-sonnet-4-20250514).
_NO_EFFORT_PREFIXES = ("claude-haiku", "claude-3", "claude-sonnet-4-5", "claude-sonnet-4-0",
                       "claude-opus-4-0", "claude-opus-4-1", "claude-sonnet-4-2", "claude-opus-4-2")
_ANTHROPIC_MAX_TOKENS = 16000  # the Messages API requires a cap
_DECLINED = "The model declined to answer that."


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic Messages API (official SDK).

    Credentials resolve the SDK's usual way (`ANTHROPIC_API_KEY`, or a saved login profile).
    Thinking is adaptive (always on for current Opus); `effort` trades depth for latency and is sent
    only to models that accept it (empty `ANTHROPIC_EFFORT` sends none). On models that support it,
    a refusal is re-run server-side on Anthropic's recommended fallback model (`fallbacks:
    "default"`). A refusal that still stands returns a fixed sentence instead of partial text; when
    streaming, words already sent can't be recalled, so the sentence is appended to mark the cut.
    """

    def __init__(self, model: str | None = None, effort: str | None = None,
                 max_tokens: int | None = None, timeout: float | None = None, client=None) -> None:
        self.model = model or DEFAULT.anthropic_model
        self.effort = effort or DEFAULT.anthropic_effort
        self.max_tokens = max_tokens or DEFAULT.llm_max_tokens or _ANTHROPIC_MAX_TOKENS
        if client is None:
            import anthropic  # noqa: PLC0415 - optional extra

            client = anthropic.Anthropic(timeout=timeout or DEFAULT.llm_timeout_s)
        self.client = client

    def _params(self, prompt: str) -> dict:
        params = {
            "model": self.model, "max_tokens": self.max_tokens, "system": _LATENCY_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.effort and self.effort != "none" and not self.model.startswith(_NO_EFFORT_PREFIXES):
            params["output_config"] = {"effort": self.effort}
        if self.model in _FALLBACK_MODELS:
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return params

    def generate(self, prompt: str, **kwargs) -> str:
        msg = self.client.beta.messages.create(**self._params(prompt))
        if msg.stop_reason == "refusal":
            return _DECLINED
        return "".join(b.text for b in msg.content if b.type == "text")

    def generate_stream(self, prompt: str, **kwargs):
        with self.client.beta.messages.stream(**self._params(prompt)) as stream:
            yielded = False
            for text in stream.text_stream:
                yielded = True
                yield text
            if stream.get_final_message().stop_reason == "refusal":
                yield (" … " if yielded else "") + _DECLINED

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("embeddings come from EMBEDDER (bge/hashing), not the LLM")


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible chat-completions API (official `openai` SDK, custom `base_url`)."""

    def __init__(self, model: str, base_url: str, api_key: str | None,
                 max_tokens: int | None = None, timeout: float | None = None, client=None) -> None:
        self.model = model
        # None = omit the cap and let the API apply its model's default (see Config.llm_max_tokens).
        self.max_tokens = max_tokens or DEFAULT.llm_max_tokens or None
        if client is None:
            from openai import OpenAI  # noqa: PLC0415 - optional extra

            # Local servers (vLLM, LM Studio) ignore the key, but the SDK requires a value.
            client = OpenAI(api_key=api_key or "unused", base_url=base_url,
                            timeout=timeout or DEFAULT.llm_timeout_s)
        self.client = client

    def _params(self, prompt: str) -> dict:
        params = {"model": self.model, "messages": [
            {"role": "system", "content": _LATENCY_SYSTEM}, {"role": "user", "content": prompt}]}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        return params

    def generate(self, prompt: str, **kwargs) -> str:
        r = self.client.chat.completions.create(**self._params(prompt))
        return r.choices[0].message.content or ""

    def generate_stream(self, prompt: str, **kwargs):
        for chunk in self.client.chat.completions.create(**self._params(prompt), stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("embeddings come from EMBEDDER (bge/hashing), not the LLM")


CLOUD_PROVIDERS = ("anthropic", "gemini", "openai")
_warned: set[str] = set()


def _warn_cloud(name: str, model: str, config: Config) -> None:
    if name in _warned:
        return
    _warned.add(name)
    print(f"WARNING: LLM_PROVIDER={name} ({model}) is a cloud API. Prompts are de-identified "
          f"(DEID_BACKEND={config.deid_backend}) before sending, but use SYNTHETIC data only, never "
          "real PHI (rules #4/#5).", file=sys.stderr, flush=True)


class DeidentifyingLLM(LLMProvider):
    """Wraps a provider so all inputs are de-identified before reaching the model (fail closed)."""

    def __init__(self, inner: LLMProvider, deidentifier: Deidentifier) -> None:
        self.inner = inner
        self.deidentifier = deidentifier

    def generate(self, prompt: str, **kwargs) -> str:
        return self.inner.generate(deidentify(self.deidentifier, prompt), **kwargs)

    def deidentify_parts(self, parts: list[tuple[str, bool]]) -> str:
        """Join `(text, sensitive)` parts, de-identifying only the sensitive ones (fail closed).

        Non-sensitive parts are fixed text the relay controls (instructions, headings, citation
        markers) and policy/SOP document passages, which carry no patient data. Scrubbing those
        corrupted them: `[P1]` became `<US_DRIVER_LICENSE>`, `critical_alarm_sop.md` became `<URL>`.
        Anything that can hold PHI (the question, history, patient reports, graph facts) is
        `sensitive=True` and goes through the de-identifier exactly as before.
        """
        return "".join(deidentify(self.deidentifier, t) if sensitive else t for t, sensitive in parts)

    def generate_parts(self, parts: list[tuple[str, bool]], **kwargs) -> str:
        return self.inner.generate(self.deidentify_parts(parts), **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        # De-identify the whole prompt up front (fail closed) before any token is generated.
        yield from self.inner.generate_stream(deidentify(self.deidentifier, prompt), **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed([deidentify(self.deidentifier, t) for t in texts])


def get_llm_provider(name: str | None = None, config: Config = DEFAULT, **kwargs) -> LLMProvider:
    """Build the configured LLM provider. `name` defaults to `config.llm_provider`."""
    name = name or config.llm_provider
    if name == "ollama":
        kwargs.setdefault("host", config.ollama_url)
        kwargs.setdefault("model", config.llm_model)
        return OllamaProvider(**kwargs)
    if name == "anthropic":
        kwargs.setdefault("model", config.anthropic_model)
        kwargs.setdefault("effort", config.anthropic_effort)
        _warn_cloud(name, kwargs["model"], config)
        return AnthropicProvider(max_tokens=config.llm_max_tokens, timeout=config.llm_timeout_s,
                                 **kwargs)
    if name in ("gemini", "openai"):
        model = kwargs.pop("model", None) or getattr(config, f"{name}_model")
        _warn_cloud(name, model, config)
        return OpenAICompatProvider(
            model=model, base_url=getattr(config, f"{name}_base_url"),
            api_key=os.environ.get(f"{name.upper()}_API_KEY"),
            max_tokens=config.llm_max_tokens, timeout=config.llm_timeout_s, **kwargs)
    if name != "echo":
        raise ValueError(f"unknown LLM_PROVIDER {name!r}: echo | ollama | anthropic | gemini | openai")
    return EchoLLM()


def build_llm(config: Config = DEFAULT) -> DeidentifyingLLM:
    """The orchestrator's LLM: the configured provider, always behind the de-id wrapper."""
    return DeidentifyingLLM(get_llm_provider(config.llm_provider, config),
                            get_deidentifier(config.deid_backend))
