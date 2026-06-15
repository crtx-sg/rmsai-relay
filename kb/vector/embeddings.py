"""Embedding backends behind one small protocol.

* `BGEEmbedder` — the real default (BAAI BGE via sentence-transformers), imported lazily.
* `HashingEmbedder` — a deterministic, offline, dependency-light fallback (hashed bag-of-words
  with L2 normalization). Keeps tests reproducible and the pipeline runnable with no model
  download; quality is lower, so it is a fallback, not the production path.

`get_embedder("auto")` prefers BGE and falls back to hashing if it is unavailable.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (offline, no model download)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324 (non-crypto use)
            vec[h % self.dim] += 1.0
        return _l2(vec).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class BGEEmbedder:
    """Real BGE embedder (sentence-transformers). Imported lazily; raises if unavailable."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True gives cosine-ready unit vectors.
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(name: str = "auto", **kwargs) -> Embedder:
    """Return an embedder. 'auto' tries BGE then falls back to hashing; 'bge'/'hashing' force one."""
    if name in ("bge", "auto"):
        try:
            return BGEEmbedder(**kwargs)
        except Exception:  # noqa: BLE001 - sentence-transformers missing or model download blocked
            if name == "bge":
                raise
    return HashingEmbedder(**({} if name == "auto" else kwargs))
