# docs

> **`docs/*.md` IS the clinical corpus.** `chunk_dir` globs this directory non-recursively for
> `*.md` (skipping this README), so **every markdown file dropped here becomes retrievable clinical
> evidence**. `production-plan.md` sat here and was being returned as the top hit for questions it
> had no business answering. Project/engineering docs belong in `docs/project/` and upload test
> fixtures in `docs/samples/`, both of which the non-recursive glob excludes; non-markdown files
> (`.html`, `.py`, and **`.pdf`** — those go through `cli.kb_upload`) are never indexed.

Small committed clinical-protocol corpus for the knowledge base. Phase 2A indexes these into the
vector store (Qdrant); Phase 2B also extracts `Condition`/`Treatment`/`Guideline` entities from the
same text into the graph, onto shared nodes. Kept deliberately small and deterministic so retrieval
and evaluation are reproducible.

`samples/` holds documents that are **not** part of that corpus: synthetic reference material for
exercising the `cli.kb_upload` path end-to-end. Because the glob is non-recursive they are only in
the KB if you upload them, which is the point — see README §3a.
