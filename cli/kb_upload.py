"""Upload documents (PDF / markdown / text) into the knowledge base.

For the reference material a unit actually runs on — clinical protocols, SOPs, guidance notes,
checklists — so questions like "what is the SOP for handling a patient with AF?" are answered from
your documents with a citation, instead of declined.

  # one file, or a whole folder
  uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf
  uv run --extra pdf python -m cli.kb_upload --dir protocols/ --glob '*.pdf'

  # see what would be indexed, without touching the store
  uv run --extra pdf python -m cli.kb_upload --file protocols/af_sop.pdf --dry-run

  # also pull entities into the graph, so the hybrid retriever can relate them
  uv run --extra pdf python -m cli.kb_upload --dir protocols/ --extract

Uploads are **incremental and idempotent**: chunk ids are content-addressed, so re-uploading an
unchanged document is a no-op and re-uploading an edited one replaces the chunks it changed. Nothing
already in the store is dropped (unlike `cli.kb_vector index --reset`, which rebuilds the corpus).

PDF pages are cited individually — an answer sourced from `af_sop.pdf#page 3` tells the clinician
which page to turn to. Scanned/image PDFs have no extractable text and are rejected with that
diagnosis rather than being indexed as empty.

⚠ An uploaded document lives **only in the vector store** — it is not copied into `docs/`. So
`cli.kb_vector index --reset`, which recreates the collection from `docs/`, drops every upload.
Keep the source files in a folder you can re-run `--dir` against after any rebuild.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.config import DEFAULT
from kb.vector.chunking import chunk_document
from kb.vector.loader import SUPPORTED_SUFFIXES, read_document
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore


def resolve_files(files: list[str], directory: str | None, glob: str) -> list[Path]:
    """Collect the documents to upload: explicit `--file`s plus everything matching `--dir/--glob`.

    Sorted and de-duplicated so a run is deterministic and passing the same file twice (directly and
    via the directory) uploads it once. Unsupported suffixes found by the glob are dropped silently —
    a folder of protocols routinely also holds spreadsheets — but an explicitly named unsupported
    file is kept, so `read_document` can tell the user why it was rejected.
    """
    picked = [Path(f) for f in files]
    if directory:
        picked += [p for p in sorted(Path(directory).glob(glob))
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    seen, out = set(), []
    for path in picked:
        key = path.resolve() if path.exists() else path
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def keep_copy(path: Path, upload_dir: Path) -> Path:
    """Copy an uploaded document into the managed folder, and return where it now lives.

    An uploaded document otherwise exists only as vectors: `cli.kb_vector index --reset` recreates
    the collection from disk and would delete it with no trace of what was lost. Keeping the source
    here makes a rebuild able to re-index it, so "rebuild the corpus" stops meaning "silently drop
    everything anyone uploaded".

    Same-name means same document, so a re-upload overwrites — that is how an edited protocol is
    updated. A file already inside the folder is left alone rather than copied onto itself.
    """
    import shutil  # noqa: PLC0415

    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / path.name
    if target.exists() and target.samefile(path):
        return target
    if target.exists():
        print(f"[upload] replacing managed copy of {path.name}", flush=True)
    shutil.copy2(path, target)
    return target


def check_embedder_matches_store(store: QdrantStore, dim: int) -> str | None:
    """Return an error message if the store's vectors were built by a different embedder.

    Uploading 384-dim vectors into a 256-dim collection is the failure this exists to prevent: it
    surfaces as an opaque client error, or worse, silently unusable search. The dimension is the
    only visible fingerprint of which embedder built a collection.
    """
    existing = store.vector_dim()
    if existing is None or existing == dim:
        return None
    return (
        f"embedder/collection mismatch: this embedder produces {dim}-dim vectors but collection "
        f"'{store.collection}' holds {existing}-dim ones. Either upload with the embedder the "
        f"corpus was built with (EMBEDDER in .env), or rebuild it: "
        f"`python -m cli.kb_vector index --dir docs --reset` (which re-indexes docs/ from scratch)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action="append", default=[],
                        help="Document to upload (repeatable).")
    parser.add_argument("--dir", default=None, help="Upload every matching file in this directory.")
    parser.add_argument("--glob", default="*", help="Pattern for --dir (default: every supported type).")
    parser.add_argument("--embedder", default=DEFAULT.embedder, choices=["auto", "bge", "hashing"],
                        help=f"Must match the corpus (default from EMBEDDER: {DEFAULT.embedder}).")
    parser.add_argument("--qdrant-url", default=DEFAULT.qdrant_url)
    parser.add_argument("--extract", action="store_true",
                        help="Also extract entities into the graph (hybrid retrieval).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be indexed; touch nothing.")
    parser.add_argument("--upload-dir", default=DEFAULT.kb_upload_dir,
                        help=f"Managed copy of uploads, re-indexed by rebuilds (default: "
                             f"{DEFAULT.kb_upload_dir}).")
    parser.add_argument("--no-keep", action="store_true",
                        help="Index in place without a managed copy (a corpus rebuild will drop it).")
    args = parser.parse_args(argv)

    paths = resolve_files(args.file, args.dir, args.glob)
    if not paths:
        parser.error("nothing to upload: pass --file and/or --dir "
                     f"(supported: {', '.join(SUPPORTED_SUFFIXES)})")

    # Read and chunk everything first. A document that cannot be read is reported and skipped rather
    # than aborting the batch — one unreadable scan should not block a folder of good protocols.
    plans, failures = [], []
    for path in paths:
        try:
            chunks = chunk_document(read_document(path), path.name)
            if not chunks:
                raise ValueError(f"{path}: produced no chunks (document has no body text)")
            plans.append((path, chunks))
        except Exception as exc:  # noqa: BLE001 - report every bad file, don't fail the batch
            failures.append((path, exc))
            print(f"[upload] SKIP {path}: {type(exc).__name__}: {exc}", flush=True)

    for path, chunks in plans:
        pages = sorted({c.source for c in chunks})
        print(f"[upload] {path.name}: {len(chunks)} chunk(s) across {len(pages)} section(s)")
        if args.dry_run:
            for chunk in chunks[:3]:
                print(f"           [{chunk.source}] {chunk.text[:90].replace(chr(10), ' ')}…")
            if len(chunks) > 3:
                print(f"           … and {len(chunks) - 3} more")

    if args.dry_run:
        print(f"[upload] dry run — nothing indexed ({sum(len(c) for _, c in plans)} chunk(s) ready)")
        return 1 if failures else 0

    retriever = VectorRetriever.build(
        store=QdrantStore.connect(args.qdrant_url), embedder_name=args.embedder
    )
    if (problem := check_embedder_matches_store(retriever.store, retriever.embedder.dim)):
        print(f"[upload] {problem}", flush=True)
        return 2

    driver = None
    if args.extract:
        from kb.graph.driver import GraphDriver  # noqa: PLC0415

        driver = GraphDriver.from_config(DEFAULT)
    try:
        total = 0
        for path, chunks in plans:
            if not args.no_keep:
                # Copy BEFORE indexing: if indexing fails the source is still preserved, which is
                # the direction that matters — a rebuild can retry, a lost file cannot.
                keep_copy(path, Path(args.upload_dir))
            # add_document re-chunks internally, so hand it the text once; the chunk plan above is
            # what we reported and (with --extract) what the graph sees, so both stay consistent.
            added = retriever.add_document(path.name, read_document(path))
            total += added
            print(f"[upload] indexed {path.name} -> {added} chunk(s) "
                  f"(embedder {args.embedder}, dim {retriever.embedder.dim})")
            if driver is not None:
                from kb.graph.extract import extract_chunk  # noqa: PLC0415

                summaries = [extract_chunk(driver, c) for c in chunks]
                conditions = {c for s in summaries for c in s["conditions"]}
                treatments = {t for s in summaries for t in s["treatments"]}
                print(f"[upload]   graph: {len(conditions)} condition(s), "
                      f"{len(treatments)} treatment(s) linked from {path.name}")
    finally:
        if driver is not None:
            driver.close()

    print(f"[upload] done: {total} chunk(s) from {len(plans)} document(s)"
          + (f"; {len(failures)} skipped" if failures else ""))
    print("[upload] try it: python -m cli.kb_vector --embedder "
          f"{args.embedder} retrieve 'what is the SOP for atrial fibrillation?'")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
