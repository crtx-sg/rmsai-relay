"""Document chunking for the vector store.

Splits markdown documents into citable chunks along section headings (and further by size if a
section is long). Each chunk carries a `source` citation of the form `filename#Heading`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_DEFAULT_MAX_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # citation, e.g. "afib_rvr.md#Rate control"
    doc_id: str  # filename
    idx: int  # position within the document


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Yield (heading, body) sections. Text before the first heading uses heading ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections]


def _split_by_size(body: str, max_chars: int) -> list[str]:
    """Split an over-long section body on paragraph boundaries, packing up to max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > max_chars:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return out


def chunk_document(
    text: str, doc_id: str, *, max_chars: int = _DEFAULT_MAX_CHARS
) -> list[Chunk]:
    """Chunk one document's text into citable `Chunk`s."""
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        if not body:
            continue
        title = heading or doc_id
        for piece in _split_by_size(body, max_chars):
            citation = f"{doc_id}#{heading}" if heading else doc_id
            # Prefix the heading into the text so the embedding sees the topic.
            chunk_text = f"{title}\n{piece}" if heading else piece
            chunks.append(Chunk(text=chunk_text, source=citation, doc_id=doc_id, idx=len(chunks)))
    return chunks


def chunk_file(path: str | Path, *, max_chars: int = _DEFAULT_MAX_CHARS) -> list[Chunk]:
    path = Path(path)
    return chunk_document(path.read_text(encoding="utf-8"), path.name, max_chars=max_chars)


def chunk_dir(directory: str | Path, *, pattern: str = "*.md", **kwargs) -> list[Chunk]:
    """Chunk every matching document in a directory (sorted for determinism)."""
    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.glob(pattern)):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_file(path, **kwargs))
    return chunks
