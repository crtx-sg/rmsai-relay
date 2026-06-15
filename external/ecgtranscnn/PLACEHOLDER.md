# Vendored: ecgtranscnn

This directory holds a **vendored clone** of an external project. Its contents are gitignored
(except this file). The ECG model **and** the synthetic-data simulator/scripts both come from here;
we **wrap** them (`inference/`, `ingest/`, generator), never reimplement.

## How to obtain

```bash
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn
```

- **Source:** https://github.com/crtx-sg/ecgtranscnn (MIT)
- **Pinned commit:** `0bc646da5409c319e75fe87eebae276d0725d096`

Prefer the clone (not `pip install git+...`) so the `scripts/` simulators and `models/` checkpoints
come with it.

## What we use

- `ecg_transcovnet/constants.py` → `CLASS_NAMES` (16 `event_type`s; source of truth).
- `ecg_transcovnet/{model,preprocessing}.py` → wrapped by `ECGModel`.
- `ecg_transcovnet/mews.py` (`calculate_mews`, `compute_mews_history`, `assess_event_trends`,
  `correlate_ecg_vitals`) → wrapped by `VitalsAnalysis`.
- `ecg_transcovnet/report.py` → per-event markdown report.
- `ecg_transcovnet/simulator/` + `scripts/generate_inference_data.py` → synthetic HDF5 into
  `data/synthetic/`.

## Model checkpoints (`models/`)

Expected checkpoint filename(s): see `external/ecgtranscnn/models/README.md` upstream. Weights may
be large / Git-LFS; a plain clone may omit them. **Until weights are present, the deterministic
`ECGModel` stub is used** — Phase 0/1 tests need no weights. If a download is blocked, place the
files manually into `external/ecgtranscnn/models/`.

> Importing the `ecg_transcovnet` package eagerly imports `torch` + `matplotlib` (via its
> `__init__`). These are installed by `uv sync`; the CPU torch wheel is used (no GPU needed).
