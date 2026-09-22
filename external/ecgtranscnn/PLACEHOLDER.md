# Vendored: ecgtranscnn

This directory holds a **vendored clone** of an external project. Its contents are gitignored
(except this file). The ECG model **and** the synthetic-data simulator/scripts both come from here;
we **wrap** them (`inference/`, `ingest/`, generator), never reimplement.

## How to obtain

```bash
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn
git -C external/ecgtranscnn checkout bac4a01
make external    # (re)install it editable
```

- **Source:** https://github.com/crtx-sg/ecgtranscnn (MIT)
- **Pinned commit:** `bac4a01`. Was `0bc646da5409c319e75fe87eebae276d0725d096`; that snapshot
  predates the real-ECG work and has no `checkpoint.py`/`classes.py`, so a 13-class checkpoint
  cannot load through it.

Prefer the clone (not `pip install git+...`) so the `scripts/` simulators come with it.

## What we use

- `ecg_transcovnet/checkpoint.py` → `load_models`, `LoadedModel`. **The load path.** It sizes the
  head from the checkpoint, builds an ensemble from several paths (softmax-averaged), and rejects
  members that disagree on head, leads or filter preset.
- `ecg_transcovnet/classes.py` → `ClassSpec`; `LoadedModel.class_spec.names` is the label list in
  output order. Read the head from there — never hard-code it.
- `ecg_transcovnet/constants.py` → `CLASS_NAMES`, the 16-name `Condition` vocabulary (mirrored in
  `common/event_types.py`, parity-tested). Any one checkpoint's head is a subset of it.
- `ecg_transcovnet/{model,preprocessing}.py` → wrapped by `ECGModel`.
- `ecg_transcovnet/mews.py` (`calculate_mews`, `compute_mews_history`, `assess_event_trends`,
  `correlate_ecg_vitals`) → wrapped by `VitalsAnalysis`.
- `ecg_transcovnet/report.py` → per-event markdown report.
- `ecg_transcovnet/simulator/` + `scripts/generate_inference_data.py` → synthetic HDF5 into
  `data/synthetic/`.

## Model checkpoints (`models/`)

`models/**/*.pt` is **gitignored upstream**, so a clone never carries weights — only the metadata
(`models/README.md`, `real_v2/cv_summary.json`, `real_v2/reports/`). Copy them in with:

```bash
make weights                                    # source defaults to ../ecgtranscnn
make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn
```

| Path | Head | Preset | Use |
|---|---|---|---|
| `models/real_v2/fold{0..4}.pt` | 13 | `default` | **Real ECG — the recommended ensemble** |
| `models/real_v2/best_model.pt` | 13 | `default` | Single-model fallback (copy of `fold0.pt`) |
| `models/best_model.pt`, `noise_robust/`, `avblock_fix/` | 16 | `none` | Simulator data only |

The simulator checkpoints score 10–26 % on real ECG, so `ECG_CHECKPOINTS` should point at the
`real_v2` folds for anything but simulator-generated HDF5. **Until weights are present the
deterministic `ECGModel` stub is used** — Phase 0/1 tests need no weights.

See `inference/README.md` for the input contract and the per-class reliability caveats, and
upstream's README § "Using This Model From Another Project" for the full brief.

> Importing the `ecg_transcovnet` package eagerly imports `torch` + `matplotlib` (via its
> `__init__`). These are installed by `uv sync`; the CPU torch wheel is used (no GPU needed).
