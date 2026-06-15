"""LLM providers + a de-identifying wrapper.

* `EchoLLM` — deterministic test provider (captures prompts, echoes the question). No network.
* `OllamaProvider` — local self-hosted LLM (POC default for real use), stdlib HTTP, lazy.
* `DeidentifyingLLM` — wraps any provider so **every** `generate`/`embed` call de-identifies its
  input first (fail closed). This is how "PHI is scrubbed before any model call" is enforced
  centrally: the orchestrator only ever holds a de-identifying provider.
"""

from __future__ import annotations

import json
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


class DeidentifyingLLM(LLMProvider):
    """Wraps a provider so all inputs are de-identified before reaching the model (fail closed)."""

    def __init__(self, inner: LLMProvider, deidentifier: Deidentifier) -> None:
        self.inner = inner
        self.deidentifier = deidentifier

    def generate(self, prompt: str, **kwargs) -> str:
        return self.inner.generate(deidentify(self.deidentifier, prompt), **kwargs)

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
    return EchoLLM()


def build_llm(config: Config = DEFAULT) -> DeidentifyingLLM:
    """The orchestrator's LLM: the configured provider, always behind the de-id wrapper."""
    return DeidentifyingLLM(get_llm_provider(config.llm_provider, config),
                            get_deidentifier(config.deid_backend))
