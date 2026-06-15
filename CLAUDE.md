# rmsai-relay — Working Agreement (do not violate)

Self-hostable POC: ingest physiological events (HDF5/MQTT) → classify arrhythmias with the
`ECG_TransConv` model → persist to a graph + vector KB → report clinically-significant events to a
remote clinician over the phone → answer follow-ups (voice/text) grounded in a clinical KB + a
per-patient knowledge graph.

## Hard rules

1. **Build leaf-up, one phase at a time.** Implement only the current phase. Each subsystem gets its
   own CLI harness and a stub for anything it depends on, is proven in isolation, then wired upward.
2. **Stop and review after every phase.** When a phase's CLI test passes, show the diff + a summary,
   then **wait for explicit go-ahead** before the next phase. Do not chain phases.
3. **No big-bang generation.** Never produce the whole codebase at once.
4. **Self-hosted by default; provider abstraction always.** All real/PHI processing runs against a
   local model server. LLM sits behind one `LLMProvider` interface (`LocalProvider` default;
   `Anthropic`/`OpenAI` swappable). Cloud APIs only on synthetic data, never PHI.
5. **Synthetic data only in development.** PHI never reaches a third-party API and is never written
   to plain-text logs.
6. **Redaction by construction.** Reference patients by `id`/`pseudonym` everywhere, including logs.
   Never log names or free-text notes.
7. **Test discipline.** Every phase ships `pytest` tests + a runnable CLI. A phase is "done" only
   when its test passes. Prefer deterministic tests with hand-labelled expected output.
8. **Resilience.** Graph traversals and RAG chains tolerate missing nodes / broken chains without
   throwing.
9. **Ask before assuming.** If a contract or design choice is ambiguous, ask rather than guess.

## Frozen contracts (defined in `common/`, do not couple to the external HDF5 layout)

- `SignalWindow` — reader output (signals, vitals+history, window geometry). **No predicted
  `event_type`.**
- `DeviceEvent` — `SignalWindow` + model-predicted `event_type`/`confidence`, FP flag, MEWS,
  vital trends, care guidance, markdown report.
- `RetrievalResult` — two labelled blocks: *Retrieved passages* (vector) + *Known relationships*
  (graph), separately cited; no cross-block re-rank.
- Interfaces: `LLMProvider`, `ECGModel` (stub until weights), `VitalsAnalysis`, `EventStore`,
  `BedAssignment`, `PatientHistory`. `event_type` ∈ the 16 `ecgtranscnn` classes; `NORMAL_SINUS`
  ⇒ false positive.

## Layout & external deps

Repo layout and the full phase plan live in `docs/`-equivalent specs (kickoff prompt + spec). The
ECG model **and** synthetic-data simulator are **vendored** from `github.com/crtx-sg/ecgtranscnn`
into `external/ecgtranscnn/` (gitignored; see `external/ecgtranscnn/PLACEHOLDER.md`) and **wrapped**,
never reimplemented. Run via `uv` (`uv run pytest`, `uv run python -m cli.<x>`).
