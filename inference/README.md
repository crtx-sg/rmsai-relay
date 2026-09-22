# inference

ECGModel (wraps ecgtranscnn classifier+preprocessing) → event_type+confidence; FP gate; VitalsAnalysis (MEWS+trend); assembles enriched DeviceEvent + markdown report. Phase 1.

## Which model

`ECG_CHECKPOINTS` (see `.env.example`) selects the checkpoint(s); unset ⇒ the deterministic
`StubECGModel`. Several paths load as **one softmax-averaging ensemble**. The recommended artifact
is upstream's real-ECG release `models/real_v2/fold{0..4}.pt` — 13 classes, `default` filter preset,
trained on ecg_sigma `ecgpkg` v2. Run `make weights` first; checkpoints are gitignored on both sides.

The **head, lead order and filter preset are read from the checkpoint** (`load_models` →
`LoadedModel.class_spec.names` / `.leads` / `.filter_preset`), never hard-coded. `EcgTransConvModel`
validates both against this repo at load time and raises on mismatch, so `get_ecg_model` falls back
to the stub loudly rather than emitting silently mislabelled predictions.

**16 is the vocabulary, 13 is the active head.** `common/event_types.CLASS_NAMES` lists all 16
`Condition` names; `real_v2`'s 13 are exactly the first 13 of them, in the same order. Everything
keyed off the vocabulary — criticality, `event_type_from_text`, de-id, the graph — stays correct;
the real model simply never emits `AV_BLOCK_2_TYPE1`, `AV_BLOCK_2_TYPE2` or `ST_ELEVATION`.

## Input contract

| Property | Value |
|---|---|
| Leads | `ingest.hdf5_reader.ECG_LEADS` = `ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX` (`vVX` = V1), in order |
| Sampling rate | 200 Hz |
| Window | 2000–2400 samples (10–12 s); `SIGNAL_LENGTH` = 2400 |
| Units | mV, **raw** — `PreprocessingPipeline` z-scores per lead; do not pre-normalize |
| Tensor | `float32`, `(batch, 7, samples)` |

`window_to_lead_matrix` zero-fills missing leads and truncates/pads to 2400. A zero-filled lead is
not "no information" to the model — it is a flat lead. Windows outside 2000–2400 samples are
untested upstream; crop or segment in the reader rather than relying on the pad.

**No abstention.** The output is a softmax over the head, single-label, probabilities summing to 1.
There is no "unknown" class: an unrecognisable window still yields a confident-looking distribution
(an all-zeros window returns `SVT` at 0.33 from the `real_v2` ensemble — a shrug looks like a
prediction). The reject option is the existing confidence gate — `fp_suppress_min_confidence`,
`low_confidence_caveat`, `outbound_min_arrhythmia_confidence` — thresholded on the max probability.

## Reliability caveats (`real_v2`, from upstream's test report)

These are **documented, not enforced** — the criticality gate treats any non-`NORMAL_SINUS`
prediction the same way it always has.

- **Trust:** `NORMAL_SINUS`, `RBBB`, `SINUS_TACHYCARDIA`, `PVC`, `ATRIAL_FIBRILLATION`,
  `SINUS_BRADYCARDIA` (F1 0.74–0.92).
- **Low confidence:** `PAC` (F1 0.55), `LBBB` (0.53, recall 0.37), `AV_BLOCK_1` (0.39, precision
  0.25 — it over-triggers on normal sinus).
- **Do not use to rule out:** `ATRIAL_FLUTTER` (recall 0.06), `SVT` (recall 0.08, AUROC 0.63 — its
  ranking is near-random), `VENTRICULAR_TACHYCARDIA` (recall 0.46). A *missing* AFL/SVT/VT
  prediction means nothing.
- **`VENTRICULAR_FIBRILLATION` is not validated here.** It scores 0.84 F1, but every VF event it was
  trained and tested on has only 1–2 genuinely measured leads; the rest are reconstructed. Not
  validated for a 7-measured-lead monitor — **do not rely on it as an alarm source.**
- **Cannot predict** `AV_BLOCK_2_TYPE1` / `AV_BLOCK_2_TYPE2` (the source annotations cannot express
  Mobitz type — permanently undetectable) or `ST_ELEVATION` (too little data).
- **Accuracy by measured leads:** 0.826 with all 7 real (our deployment case), 0.727 with ECG2 only,
  0.659 with ECG2+V1.
- **Paced patients:** 0.60 accuracy vs 0.79 unpaced, specifically paced atrial fibrillation (recall
  0.20); paced PVC is unaffected. The model never saw a paced beat in training.

Research model trained on one package of public datasets. **Not a medical device** — not for
diagnosis or unsupervised alarms.
