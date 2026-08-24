"""Read a document off disk into text the chunker can cite — markdown/text natively, PDF via pypdf.

Clinical protocols, SOPs, guidance and checklists arrive as PDFs at least as often as markdown, and
a PDF is not just "text in a different container": extraction yields one string per page, hard-wrapped
at the layout width, with no blank lines between paragraphs. Handed to the chunker as-is that becomes
a single unsplittable blob per page (`_split_by_size` breaks on blank lines), which embeds badly —
one vector covering an entire page answers nothing precisely.

So PDF text is rebuilt before chunking: wrapped lines are re-joined into logical lines, each page
becomes a `## page N` section, and the chunker's existing heading logic then produces chunks that
cite `af_protocol.pdf#page 3`. A clinician checking an answer gets a page number to turn to.
"""

from __future__ import annotations

import re
from pathlib import Path

#: What `cli.kb_upload` accepts. Markdown and text are read as-is (their structure is already the
#: chunker's input format); PDF goes through extraction.
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")

_SENTENCE_END = re.compile(r"[.!?:;)\]\"']$")


def dewrap(text: str) -> str:
    """Re-join lines a PDF broke at the layout width, then blank-line-separate what remains.

    Two signals that a line break was the layout's and not the author's: the line does not end at a
    sentence boundary, and the next line continues in lower case. Requiring both keeps numbered
    steps and checklist items intact ("1. Confirm rhythm." ends in '.', so it stays its own line) —
    which matters, because splitting a checklist mid-item is worse than not splitting it at all.
    A trailing hyphen is a word broken across lines, so it re-joins without a space.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        if out and not _SENTENCE_END.search(out[-1]) and line[:1].islower():
            if out[-1].endswith("-"):
                out[-1] = out[-1][:-1] + line  # hyphenated word split across lines
            else:
                out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    # Blank lines between logical lines: the chunker packs paragraphs up to its size budget, so this
    # is what lets a long page split at all, and it keeps each checklist item individually packable.
    return "\n\n".join(out)


def pages_to_markdown(pages: list[str]) -> str:
    """PDF pages -> markdown with one `## page N` heading each, so chunks carry a page citation.

    Pages that extract to nothing (images, blank separators) are skipped but still consume their
    number, so the citation always matches the page number printed on the document.
    """
    sections = []
    for number, raw in enumerate(pages, start=1):
        body = dewrap(raw or "")
        if body:
            sections.append(f"## page {number}\n\n{body}")
    return "\n\n".join(sections)


def read_pdf(path: str | Path) -> str:
    """Extract a PDF to markdown. Raises if the `pdf` extra is missing or nothing is extractable."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install profile
        raise RuntimeError(
            "PDF support needs the `pdf` extra: uv sync --extra pdf (or `uv run --extra pdf …`)"
        ) from exc

    reader = PdfReader(str(path))
    text = pages_to_markdown([page.extract_text() or "" for page in reader.pages])
    if not text.strip():
        raise ValueError(
            f"{path}: no extractable text in {len(reader.pages)} page(s). This is almost certainly "
            f"a scanned/image PDF — OCR is not supported. Re-export it as a text PDF, or paste the "
            f"content into a markdown file and upload that."
        )
    return text


def read_document(path: str | Path) -> str:
    """Read any supported document into chunker-ready text. Raises with a legible reason."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: not a file")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"{path}: unsupported type {suffix!r} — expected one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".pdf":
        return read_pdf(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path}: file is empty")
    return text  # markdown/text is already the chunker's input format — do not reflow it
