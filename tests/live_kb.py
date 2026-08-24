"""Helper for infra tests that drive a CLI against the *deployed* vector store.

Most tests isolate Qdrant (`QdrantStore.in_memory()` or a unique collection). The full-loop CLI
tests can't: `cli.outbound` / `cli.text_chat` connect to the deployed `rmsai_docs`. Those
collections are built by whichever embedder the deployment configured (`EMBEDDER` in `.env`), but
the test process is hermetic (`RMSAI_NO_DOTENV=1` in conftest), so `DEFAULT.embedder` is the
built-in fallback and may not match. Appending with a mismatched embedder is refused, so pick the
embedder from the collection itself — its vector dimension is the only fingerprint of which one
built it (the same reasoning as `cli.kb_upload.check_embedder_matches_store`).
"""

from __future__ import annotations

import pytest

from common.config import DEFAULT

_DOCS = "rmsai_docs"


def docs_embedder_or_skip(collection: str = _DOCS) -> str:
    """Return the embedder name matching the deployed collection's vectors, or skip the test."""
    from kb.vector.embeddings import get_embedder  # noqa: PLC0415
    from kb.vector.store import QdrantStore  # noqa: PLC0415

    try:
        dim = QdrantStore.connect(DEFAULT.qdrant_url, collection).vector_dim()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qdrant unreachable: {exc}")
    if dim is None:  # collection absent: it gets created by this run, so anything goes
        return DEFAULT.embedder
    for name in ("hashing", "bge"):
        try:
            if get_embedder(name).dim == dim:
                return name
        except Exception:  # noqa: BLE001, PERF203 - backend unavailable (no sentence-transformers)
            continue
    pytest.skip(f"no available embedder produces {dim}-dim vectors for collection '{collection}'")
