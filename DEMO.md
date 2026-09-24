# Demo runbook — real ECG through the relay

End to end, in Docker: held-out **real** ECG recordings → `real_v2` classifier → event bus →
graph + vector KB → clinician worklist, voice and chat → grounded follow-up Q&A.

This is the demo path. The README has the background: [Setup](README.md#setup),
[Operations runbook](README.md#operations-runbook), and
[End-to-end testing](README.md#end-to-end-testing). For the synthetic (simulator) variant, see
[§ Synthetic variant](#synthetic-variant) at the bottom.

> Everything below runs from the repo root. Define this once per shell:
>
> ```bash
> export RMSAI="docker compose -f infra/docker-compose.yml run --rm tools python -m"
> ```

---

## 0. One-time setup

| Need | How | Check |
|---|---|---|
| Vendored model + simulator | `git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn && git -C external/ecgtranscnn checkout bac4a01` | `ls external/ecgtranscnn/ecg_transcovnet/checkpoint.py` |
| `real_v2` weights (gitignored upstream) | `make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn` | `ls external/ecgtranscnn/models/real_v2/fold{0..4}.pt` |
| ecg_sigma training package | a sibling checkout at `../ecg_sigma/packages/ecg_pkg_v2`, or set `ECGPKG_HOST_DIR` to your `packages/` dir | `ls ../ecg_sigma/packages/ecg_pkg_v2/{package.json,manifest.csv}` |
| `.env` | `cp .env.example .env`, then set `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` / `HOSPITAL_ID` **and** `ECG_CHECKPOINTS` (below) | `grep ^ECG_CHECKPOINTS .env` |
| App image | `make docker-build` (~5–10 min) | see [§1](#1-bring-up-and-confirm-the-real-model-is-loaded) |

`ECG_CHECKPOINTS` in `.env` is the real_v2 five-fold ensemble:

```bash
ECG_CHECKPOINTS=external/ecgtranscnn/models/real_v2/fold0.pt external/ecgtranscnn/models/real_v2/fold1.pt external/ecgtranscnn/models/real_v2/fold2.pt external/ecgtranscnn/models/real_v2/fold3.pt external/ecgtranscnn/models/real_v2/fold4.pt
```

> **Containers read `.env`, not your shell.** Exporting `ECG_CHECKPOINTS` on the host does nothing
> inside Docker.
>
> **Rebuild after the vendored pin moves.** The image installs `ecg_transcovnet` non-editable at
> build time. An image built before the pin reached `bac4a01` has no `ecg_transcovnet.checkpoint`,
> can't load real_v2, and **silently runs the stub**. `make docker-build`, then `make docker-up`.

---

## 1. Bring up and confirm the real model is loaded

```bash
make docker-up        # redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway
make docker-ps        # all Up; gateway "Up (healthy)"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/     # 200
```

**Confirm it is real_v2 and not the stub.** Do this before every demo; a stub fallback looks just
like a bad model:

```bash
docker compose -f infra/docker-compose.yml run --rm tools python -c \
  "from common.config import DEFAULT; from inference.ecg_model import get_ecg_model; \
   print(type(get_ecg_model(DEFAULT.ecg_checkpoints)).__name__)"
```

```
INFO rmsai.inference.ecg: ECG model loaded: 5 checkpoint(s), 13 classes, filter_preset=default, package=v2 manifest=096bdfbafa04
EcgTransConvModel
```

| You see | Meaning | Fix |
|---|---|---|
| `5 checkpoint(s), 13 classes` + `EcgTransConvModel` | real_v2 ensemble ✔ | — |
| `16 classes` | a simulator checkpoint | point `ECG_CHECKPOINTS` at `real_v2/fold*.pt` |
| `StubECGModel`, no load line | `ECG_CHECKPOINTS` unset | add it to `.env`, then `make docker-up` |
| `failed to load ECG checkpoint … No module named 'ecg_transcovnet.checkpoint'` | stale image | `make docker-build && make docker-up` |
| `ECG checkpoint(s) not found` | weights missing | `make weights ECGTRANSCNN_DIR=…` |

---

## 2. Initialize the knowledge base

Both stores start empty. They are also wiped by some tests (see [Troubleshooting](#troubleshooting)).

```bash
$RMSAI cli.graph migrate                                   # {"migrated": true}
$RMSAI cli.graph protocols                                 # {"protocols_loaded": 2}
$RMSAI cli.graph extract --dir docs                        # doc entities onto shared nodes
$RMSAI cli.kb_vector --embedder bge index --dir docs       # clinical corpus -> Qdrant
$RMSAI cli.kb_upload --file docs/samples/critical_alarm_sop.md   # optional: a sample SOP to cite
```

**Order matters:** index the documents *before* you ingest events. `cli.kb_vector index --reset`
wipes the per-event report narratives.

---

## 3. Curate a held-out real-ECG set (`real-samples`)

A converted ecg_sigma record is hundreds of 12 s windows. Most of them sit in the split real_v2 was
**trained** on, and many are labelled `OTHER`. Ingesting one whole record floods the worklist with
alerts you can't score. `cli.real_samples` picks a small, labelled, **held-out** set instead.

### See what is available

```bash
docker compose -f infra/docker-compose.yml run --rm real-samples list
```

```
package v2 (manifest 096bdfbafa04), split=test
  NORMAL_SINUS                 1400  (incart:500, mitbih:200, ptbxl:700)
  ATRIAL_FIBRILLATION           428  (afdb:200, incart:21, mitbih:95, ptbxl:112)
  PVC                           700  (incart:480, mitbih:170, ptbxl:50)
  VENTRICULAR_TACHYCARDIA        78  (incart:17, mitbih:8, vfdb:53)
  VENTRICULAR_FIBRILLATION       83  (cudb:23, mitbih:12, vfdb:48)
  …
```

### Pick events

```bash
rm -rf data/real     # optional: start clean (pick replaces same-named files but never clears the folder)
docker compose -f infra/docker-compose.yml run --rm real-samples pick \
    --pick VENTRICULAR_TACHYCARDIA:2,ATRIAL_FIBRILLATION:2,NORMAL_SINUS:1
```

```
{"label": "VENTRICULAR_TACHYCARDIA", "dataset": "incart", "subject": "incart:p2", "event_key": "event_1082", "file": "data/incart/I05_2025-01.h5"}
…
{"events": 5, "files": ["data/real/06995_2025-01.h5", "data/real/106_2025-01.h5", …], "package": "v2", "split": "test", "seed": 42}
```

### How the `pick` command works

```
docker compose -f infra/docker-compose.yml run --rm real-samples  pick --pick VENTRICULAR_TACHYCARDIA:2,ATRIAL_FIBRILLATION:2
└──────────────────── Docker ────────────────────┘ └─ service ─┘  └────────── cli.real_samples arguments ──────────┘
```

**Docker part**

- `-f infra/docker-compose.yml`: where the service definitions live.
- `run --rm real-samples`: a one-off container from the `real-samples` service, removed on exit.
  It uses the same image as the app services, plus two mounts:
  - the repo at `/app`, so output under `data/real/` appears on the host;
  - `${ECGPKG_HOST_DIR:-../ecg_sigma/packages}` at `/ecgpkg`, **read-only**.
- The service's entrypoint is `python -m cli.real_samples`, so everything after the service name is
  passed straight to the CLI. With no arguments it runs `list`.
- It is a separate service, not `tools`, so a host without ecg_sigma keeps a working `tools`. Here
  a missing package directory fails loudly instead of Docker creating an empty one.

**CLI part**

- `pick` writes files; `list` only reports.
- `--pick LABEL:N[,LABEL:N…]`: full class names, as printed by `list`. `VT:2` is rejected. A count
  must be ≥ 1, and a bare `LABEL` means 1.

**What `pick` does, step by step**

1. **Reads** `package.json` and `manifest.csv` from `/ecgpkg/ecg_pkg_v2`. No signal data is loaded
   yet.
2. **Checks the package against the model.** It compares the manifest SHA with the one recorded in
   the configured checkpoint. "Held out" only means something relative to the package the model
   was trained on, so a mismatch stops with exit code 2 unless `--allow-mismatch` is passed. With
   no checkpoint configured it warns that the check could not run.
3. **Selects** events:
   - from the **test split** only, i.e. subjects the model never saw in training;
   - deterministically: candidates are sorted, then sampled with `--seed` (default 42), so the
     same command always yields the same events;
   - a class with too few events returns what exists and prints a shortfall warning.
4. **Copies** each chosen `event_*` group out of its source recording, one output file per
   recording (`data/real/<record>.h5`). A file carries a single `patient_id`, so subjects are never
   merged. Signals are copied byte-for-byte; no conversion is reimplemented.
5. **Stamps** labels and provenance:
   - The manifest label goes on each event's `condition` attr. This becomes the reader's ground
     truth, which `cli.ingest` scores against.
   - ecg_sigma's raw annotation is kept as `source_condition`.
   - `/metadata` gets `ecgpkg_version`, `ecgpkg_manifest_sha256` and `ecgpkg_split`.
6. **Prints** one JSON line per event, then a summary line with the files written.

**Options**

| Flag | Default | Purpose |
|---|---|---|
| `--dataset mitbih vfdb …` | all | restrict to source databases |
| `--seed N` | 42 | a different, still reproducible, selection |
| `--out DIR` | `data/real` | output directory (must be outside the package; it is checksummed) |
| `--split train\|val\|test` | `test` | non-test splits work but warn: the model has seen those events |
| `--checkpoint …` | `ECG_CHECKPOINTS` | checkpoint(s) to check the package against |
| `--allow-mismatch` | off | proceed despite a package/model mismatch |
| `--package DIR` (before `pick`) | `ECGPKG_DIR` | package root; the service sets it to `/ecgpkg/${ECGPKG_NAME:-ecg_pkg_v2}` |

**Suggested demo picks**

```bash
# critical-heavy: exercises the call/alert path
--pick VENTRICULAR_TACHYCARDIA:2,VENTRICULAR_FIBRILLATION:1,ATRIAL_FIBRILLATION:2,NORMAL_SINUS:1
# a single VT for a focused walkthrough
--pick VENTRICULAR_TACHYCARDIA:1 --dataset incart
```

---

## 4. Dry-run the set (optional, recommended)

Classify without publishing, and check the score before anyone is watching:

```bash
$RMSAI cli.ingest --dir data/real
```

```
{"patient": "I05", "event_type": "VENTRICULAR_TACHYCARDIA", "confidence": 0.497, "criticality": "Critical", "ground_truth": "VENTRICULAR_TACHYCARDIA", …}
{"patient": "106", "event_type": "NORMAL_SINUS", "confidence": 0.577, "criticality": "Low", "ground_truth": "NORMAL_SINUS", …}
{"summary": {"events": 5, "scored": 5, "correct": 5, "accuracy": 1.0, "model": "checkpoint"}}
```

`"model": "stub"` plus a `WARNING … STUB` line means [§1](#1-bring-up-and-confirm-the-real-model-is-loaded)
did not pass. Stop and fix it. If you'd rather show a set with a miss or two (VT↔VF and AFib↔SVT
are the usual confusions), re-pick with another `--seed`.

---

## 5. Drive the demo

### Publish to the bus

```bash
$RMSAI cli.ingest --dir data/real --emit bus
```

### Watch the consumer

```bash
docker compose -f infra/docker-compose.yml logs -f consumer
# [consume] received event … type=VENTRICULAR_TACHYCARDIA conf=0.50 patient=I05
# [consume] persisted MonitoredEvent … -> Neo4j graph
# [consume] archived report narrative -> Qdrant vector store
# [consume] dispatch=app: pushed inbox event … -> rmsai-inbox-h1
```

Critical/High events are dispatched. `NORMAL_SINUS`/Low events are persisted but skipped with a
reason (`below_threshold`, …).

### A. Companion app (default, `DISPATCH_MODE=app`)

1. Open `http://localhost:8080/` and enter the PIN (`INBOUND_AUTH_PIN`, default `1234`).
2. The worklist fills live. Patients appear as their source record ids (`I05`, `207`,
   `PTBXL-14628`).
3. Select a row. Chat is scoped to that event, and its summary is spoken (`INBOX_SPEAK_ON_SELECT`).
4. Ask follow-ups, typed or by voice:
   - *"what were the vitals at the time of the event?"*
   - *"what is the escalation checklist for a critical alarm?"* (cites the sample SOP from §2)
   - *"which conditions are co-morbid with atrial fibrillation?"* (hybrid: passages + graph)

### B. Outbound voice call over WebRTC (no phone)

Set `DISPATCH_MODE=app+call` in `.env`, run `make docker-up` to recreate the consumer, and publish
again. For each critical event, the consumer log prints an `rmsai-outbound-<event_id>` room and a
`Token:`. Join it at <https://agents-playground.livekit.io> (Manual → URL + token → allow mic):

1. Say the PIN.
2. Hear **this** event's alert.
3. Ask follow-ups with the wake word: *"hey vios, what were the vitals?"*
4. Say *"acknowledge"* to flip the event's status.

Full detail: [README §5](README.md#5-real-webrtc-audio-loop-browser-no-phone). A real SIP call
needs a trunk and `OUTBOUND_ENABLED=true`: [README §6](README.md#6-real-sip-phone-call-outbound-to-a-number).

### Inspect what was stored

```bash
$RMSAI cli.kb_dump --list              # recent event ids
$RMSAI cli.kb_dump <event_id>          # graph node + report file + vector chunks, side by side
$RMSAI cli.kb "which critical events happened in the last hour"
```

---

## 6. Shut down

```bash
make docker-down      # stops everything; named volumes (neo4j/qdrant data, model cache) survive
```

---

## What to say about the results (caveats)

- **A handful of events is a demo, not an evaluation.** For performance, cite upstream
  `external/ecgtranscnn/models/real_v2/reports/test.md` (test accuracy 0.782, primary macro-F1
  0.587). Several classes are not safe to rule out on, and VF is not validated as an alarm source
  (see `inference/README.md`).
- **Only the ECG and HR are real.** Per ecg_sigma's README:
  - PPG, RESP and RespRate are *derived* from the ECG.
  - Pulse, SpO2, BP and Temp are *invented*, SpO2 and BP from **condition-keyed** ranges. So MEWS
    and vitals-driven criticality partly echo the label, not the patient.
  - On MIT-BIH, only leads II and V1 are recorded; the other limb leads are computed from them.
- **Never use the train split for numbers.** For example, MIT-BIH record 105 scores 0.780, but it
  is in ecgpkg v2's *train* split. `pick` defaults to `test` for this reason.
- **The data is public, de-identified research data** (PhysioNet). Patient ids are record ids, not
  people.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[real_samples] ../ecg_sigma/packages/ecg_pkg_v2 is not an ecgpkg` | run via `tools`, which can't see `../ecg_sigma` | use the `real-samples` service |
| `real-samples` fails to start: bind source path does not exist | package not at `../ecg_sigma/packages` | set `ECGPKG_HOST_DIR=/abs/path/to/packages` (and `ECGPKG_NAME` if not `ecg_pkg_v2`) |
| `pick` exits 2: `… not held out for this model` | package ≠ checkpoint's training package | use the matching package, or `--allow-mismatch` knowingly |
| Accuracy ≈ 0, `ST_ELEVATION` predictions, `"model": "stub"` | stub fallback | [§1 table](#1-bring-up-and-confirm-the-real-model-is-loaded) |
| Worklist empty / `cli.kb_dump --list` returns `[]` | `tests/test_graph_templates.py`, `tests/test_orchestrator.py`, `cli.kb_eval` reset the **live** Neo4j | redo §2, then republish (§5) |
| In-app chat gets no reply | voice worker restarted after the app connected | `$RMSAI cli.dispatch --all-inbox` |
| Voice worker loops on `:7880` | LiveKit container not running | `docker compose -f infra/docker-compose.yml up -d livekit` |
| Two consumers splitting events | a host-run `cli.consume` shares group `rmsai.relay` with the container | `docker compose -f infra/docker-compose.yml stop consumer` before running one by hand |
| Code edit not picked up | — | `make docker-restart` (source is bind-mounted); rebuild only for dependency / vendored changes |

---

## Synthetic variant

Same flow, with simulator output instead of real recordings. It is useful offline, or without
ecg_sigma, and it exercises the plumbing, **not** the model. real_v2 scores 10–26 % on simulator
data. For plausible labels, pass
`--checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt`:

```bash
docker compose -f infra/docker-compose.yml run --rm tools \
    python external/ecgtranscnn/scripts/generate_inference_data.py --output-dir data/inference
$RMSAI cli.ingest --file data/inference/<file>.h5 --emit bus
```
