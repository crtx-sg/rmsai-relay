# Operations

Running the relay: the lifecycle runbook, what you can ask the knowledge base, and how model performance is traced. Install first: [INSTALL.md](INSTALL.md).

## Operations runbook

The whole lifecycle in order — bring the stack up, load the model, initialize the KB, drive it,
inspect it, measure it, shut it down. Every command here is Docker-first; the host equivalent of
any `$RMSAI` line is the same `python -m …` under `uv run`.

Define this once per shell — it runs any CLI harness in the app image with the repo mounted, so no
host Python is needed:

```bash
export RMSAI="docker compose -f infra/docker-compose.yml run --rm tools python -m"
$RMSAI cli.kb_dump --list          # …and so on for every cli.* below
```

### 1. Prerequisites and the vendored ECG model

The `ECG_TransConv` classifier and the synthetic-data simulator are **vendored, not vendorable by
pip** — the clone carries the `scripts/` simulators and the `models/` metadata that a wheel would
not. It is gitignored but **baked into the image**, so clone it *before* the first build:

```bash
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn   # pinned: a843ccf
git -C external/ecgtranscnn checkout a843ccf
cp .env.example .env    # then set LIVEKIT_API_KEY / LIVEKIT_API_SECRET / HOSPITAL_ID
```

**The clone carries no weights.** `models/**/*.pt` is gitignored upstream, so this is expected, not
a failed download. Copy them in from a local ecgtranscnn working copy that has them:

```bash
make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn
ls external/ecgtranscnn/models/real_v2/*.pt
# external/ecgtranscnn/models/real_v2/best_model.pt   (single-model fallback)
# external/ecgtranscnn/models/real_v2/fold0.pt … fold4.pt
```

Then point `ECG_CHECKPOINTS` at the five folds (see `.env.example`), or pass them per-run as
`--checkpoint` to `cli.ingest` / `cli.outbound` (below).

**Without weights the pipeline still runs** as long as `ECG_CHECKPOINTS` is unset: `ECGModel` is then
the deterministic stub, so every phase and test works and only the predictions are synthetic. A
configured checkpoint that is missing or unloadable stops `cli.ingest`/`cli.outbound` with an error;
pass `--checkpoint` with no value to run the stub on purpose.

**Which checkpoint.** `real_v2` is the real-ECG release (13 classes, 5-fold ensemble, trained on
ecg_sigma `ecgpkg` v2) and is what you want for real recordings. The older simulator checkpoints
(`models/best_model.pt`, `noise_robust/`, `avblock_fix/`; 16 classes) score 10–26 % on real ECG —
use them only against simulator-generated HDF5. The head, lead order and filter preset travel
inside the checkpoint, so switching between them needs no code change. See `inference/README.md`
for the input contract and the per-class reliability caveats — several classes are **not** safe to
rule out on, and `VENTRICULAR_FIBRILLATION` is not validated as an alarm source.

Generate synthetic HDF5 to feed the pipeline. The generator's `--output-dir` defaults to
`data/inference` **relative to the current directory**, so run it from the repo root (or pass the
flag) — otherwise the files land somewhere you then can't find, e.g. under
`external/ecgtranscnn/data/inference/` if you `cd` in there first:

```bash
# a script rather than a module, so it does not use $RMSAI
docker compose -f infra/docker-compose.yml run --rm tools \
    python external/ecgtranscnn/scripts/generate_inference_data.py --output-dir data/inference
ls data/inference/*.h5
```

Useful flags: `--num-files` / `--events-per-file` (default 3 × 5), `--conditions
ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1` to weight the mix, and `--noise-level {low,medium,high,mixed}`.
`data/` is gitignored and bind-mounted into every container, so these files are visible to both the
host CLIs and the `tools` service at the same path.

#### Feeding it real ECG (what `real_v2` is actually for)

The generator above produces **simulator** output. `real_v2` is trained on real recordings and
scores 10–26 % on synthetic data, so to exercise it meaningfully you need real-ECG HDF5 — a file
derived from an ecg_sigma `ecgpkg` package, carrying the same event/lead layout.

Two `/metadata` layouts are in circulation and the reader accepts **both**: the simulator writes
each field as a dataset, ecg_sigma-derived files write them as attributes. No conversion needed.

```bash
ECG_CHECKPOINTS="$(ls external/ecgtranscnn/models/real_v2/fold[0-4].pt | tr '\n' ' ')" \
uv run python -m cli.ingest --file data/inference/<real>.h5
```

```
INFO rmsai.inference.ecg: ECG model loaded: 5 checkpoint(s), 13 classes,
     filter_preset=default, package=v2 manifest=096bdfbafa04
{"patient": "105", "event_id": "8b9c551a…", "event_type": "PVC", "confidence": 0.823,
 "false_positive": false, "criticality": "High", "mews": 1, "ground_truth": "PVC", "outcome": "TP"}
```

The startup line is the check that matters: **13 classes** and **`filter_preset=default`** confirm
the real checkpoint loaded. A 16-class line means a simulator checkpoint; no line at all means the
stub (no `ECG_CHECKPOINTS`, or `--checkpoint` with no value). A configured checkpoint that is missing
or fails to load stops `cli.ingest` with an error instead of falling back.

> **Only for `--emit stdout`.** A raw ecg_sigma file keeps the dataset's record id as the patient
> (`"105"` above). Published with `--emit bus` while the hospital's `dispatch_mode` includes `app`, the consumer
> persists it, then fails on the worklist push (`[retry] … refusing to publish non-pseudonym patient
> ref: '105'`); it retries every `--retry-idle-ms` (10 min) and after `--max-attempts` (3) moves the
> message to `rmsai.events.dead`. The event is persisted but never dispatched. For anything that goes on the bus, curate with `cli.real_samples` below,
> which assigns `PT9#####` pseudonyms.

> **Sanity-check your own files before trusting a number.** Running the 51-event MIT-BIH record 105
> file through this path scores 0.780 — but `mitbih:105` is in `ecgpkg` v2's **train** split, so
> that figure is measured on data the model was fitted to and is *not* a performance estimate. It
> shows the pipeline works end to end on real ECG, nothing more. For a real number, use held-out
> subjects and read `models/real_v2/reports/test.md` upstream (test accuracy 0.782, primary
> macro-F1 0.587).

##### Curating a held-out real-ECG demo set (`cli.real_samples`)

> The full demo walkthrough, including every `pick` flag, lives in [`DEMO.md`](DEMO.md#3-curate-a-held-out-real-ecg-set-real-samples).

A converted record is hundreds of windows, most of them in the split `real_v2` was **trained** on,
and many carry no usable label. `cli.real_samples` picks a small, labelled, **held-out** set out of
the ecg_sigma training package instead and writes it as reader-ready HDF5:

```bash
# which labels the package's test split can supply (default package: ECGPKG_DIR, else
# ../ecg_sigma/packages/ecg_pkg_v2)
uv run python -m cli.real_samples list

# pick events by class; test split only, deterministic for --seed; writes data/real/<record>.h5
uv run python -m cli.real_samples pick \
    --pick VENTRICULAR_TACHYCARDIA:3,VENTRICULAR_FIBRILLATION:2,ATRIAL_FIBRILLATION:3,PVC:2,NORMAL_SINUS:2

# feed the whole set: --dir ingests every *.h5; add --emit bus to drive the consumer/app/call path
uv run python -m cli.ingest --dir data/real
```

```
{"patient": "PT992591", "event_type": "VENTRICULAR_TACHYCARDIA", "confidence": 0.497, "criticality": "Critical", "ground_truth": "VENTRICULAR_TACHYCARDIA", …}
{"patient": "PT952625", "event_type": "SVT", "confidence": 0.538, "criticality": "High", "ground_truth": "ATRIAL_FIBRILLATION", …}
{"summary": {"events": 12, "scored": 12, "correct": 9, "accuracy": 0.75, …, "model": "checkpoint"}}     # stderr
```

- **One output file per source record.** A file carries one `patient_id`, so subjects are never
  merged. The patient is a stable `PT9#####` pseudonym derived from the subject, never the record
  id, because the app inbox refuses non-pseudonyms. The source id is kept on `/metadata`.
- **Ground truth is the manifest label**, written to each event's `condition` attr (ecg_sigma's own
  raw annotation is kept as `source_condition`). `cli.ingest` prints the scored summary to stderr
  whenever events carry a known class, simulator files included.
- **Package ↔ model check.** "Held out" only means something relative to the package the checkpoint
  was trained on, so `pick` compares the package manifest with the one recorded in
  `ECG_CHECKPOINTS` (or `--checkpoint`) and refuses a mismatch unless `--allow-mismatch`.
  `--split train|val` works but warns that the numbers are not a performance estimate.
- **A 12-event set is a demo, not an evaluation.** For performance use upstream
  `models/real_v2/reports/test.md`.
- **Docker:** `tools` does not see `../ecg_sigma`, so use the `real-samples` service. It mounts
  `ECGPKG_HOST_DIR` (default: the sibling `ecg_sigma/packages`) read-only at `/ecgpkg`, uses
  `ECGPKG_NAME` (default `ecg_pkg_v2`) inside it, and fails loudly if the folder is missing. Both
  are Compose settings: export them in the shell (an absolute path is safest), not in `.env`.
  Output lands in the mounted `data/`:

  ```bash
  docker compose -f infra/docker-compose.yml run --rm real-samples list
  docker compose -f infra/docker-compose.yml run --rm real-samples pick --pick VENTRICULAR_TACHYCARDIA:2,ATRIAL_FIBRILLATION:2
  $RMSAI cli.ingest --dir data/real --emit bus
  ```

  Two things must hold:
  1. `ECG_CHECKPOINTS` is set in `.env` (the containers read `.env`, not your shell); if it is
     unset, every prediction comes from the **stub**, flagged by `"model": "stub"` plus a WARNING.
  2. The image was built **after** the vendored pin moved to `bac4a01`. An older image lacks
     `ecg_transcovnet.checkpoint`, so `cli.ingest` stops with `failed to load ECG checkpoint: …`. Fix it with
     `make docker-build`, then `make docker-up` so running services pick up the new image.
- **Only the ECG (and HR) is real.** Per ecg_sigma's README: PPG/RESP/RespRate are derived from the
  ECG, and Pulse/SpO2/BP/Temp are invented, SpO2/BP from *condition-keyed* ranges. So MEWS and
  vitals-driven criticality on these events partly echo the label rather than the patient. On
  MIT-BIH only leads II and V1 are recorded; the other limb leads are computed from them.

### 2. Build and start

```bash
make docker-build      # once; ~5-10 min (torch, faster-whisper, presidio, spaCy)
make docker-up         # redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway
make demo-check        # or `make demo-up` for both: ordered start + health check
make docker-ps
```

Wait for all three app services to report ready — cold start loads BGE/Whisper/the orchestrator:

```bash
docker compose -f infra/docker-compose.yml logs consumer     | grep "\[consume\] group="
docker compose -f infra/docker-compose.yml logs voice-worker | grep "registered worker"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/     # 200
```

Expected steady state:

```
consumer       Up  (blocking on XREADGROUP)
gateway        Up (healthy)   → http://localhost:8080/
voice-worker   Up (healthy)   (registered as rmsai-agent, health server on 8091)
livekit neo4j qdrant redis   Up (healthy)
```

> **Before `make docker-up`, stop any host-run `cli.gateway` / `cli.voice_worker`.** The app
> services use host networking, so a hand-run gateway holds 8080 and a hand-run worker holds 8081 —
> the containers then fail to bind. Two workers registered under the same `LIVEKIT_AGENT_NAME` also
> split dispatches between them.

### 3. Initialize the knowledge base

Both stores start empty. The **graph** needs its schema, care protocols, and document entities; the
**vector** store needs the clinical corpus. `cli.consume` runs `migrate` and appends `docs/` to Qdrant
itself on startup, so strictly only `protocols` and `extract` are manual:

```bash
$RMSAI cli.graph migrate                     # constraints + indexes   -> {"migrated": true}
$RMSAI cli.graph protocols                   # care protocols          -> {"protocols_loaded": 2}
$RMSAI cli.graph extract --dir docs          # doc entities onto shared nodes
#   -> {"chunks": 13, "guidelines": 13, "conditions": 5, "treatments": 9}
$RMSAI cli.kb_vector --embedder bge index --dir docs    # clinical corpus -> Qdrant
```

Optionally seed a synthetic patient cohort (demographics, co-morbidities, symptoms) so the
pattern/co-morbidity queries have something to traverse. Patients are auto-created on first event
otherwise (G8):

```bash
$RMSAI cli.graph ingest --patients PT1000 PT1001 PT1002
```

**The embedder must match the collection** (hashing=256-dim / BGE=384-dim). Every KB path defaults
`--embedder` to `EMBEDDER` from `.env`, except `cli.kb_vector` and `cli.kb` (default `auto`) and
`cli.kb_eval` (default `hashing`), so pass it explicitly there. See [§3a](TESTING.md#3a-upload-protocols--sops--checklists-to-the-kb).

### 4. Drive the application

**Open the app** at `http://localhost:8080/` (PIN `INBOUND_AUTH_PIN`, default `1234`). On login
the worklist loads every **pending** event from the graph, so events published while the app was
closed still appear. A live push to a closed app still fails (the consumer logs `inbox push failed …
503 … inbox room not open`), but that event shows up on the next login. Every stored event,
including acknowledged and withheld ones, is in the **Events** tab.

Publish classified events onto the bus; the running `consumer` container picks them up, persists to
both stores, and dispatches per the criticality gate:

```bash
$RMSAI cli.ingest \
    --file data/inference/PT6580_2026-06.h5 --emit bus
# uses ECG_CHECKPOINTS from .env; override per-run with
#   --checkpoint external/ecgtranscnn/models/real_v2/fold{0,1,2,3,4}.pt
# a missing/unloadable checkpoint stops ingest with an error; omit both, or pass --checkpoint with no value, to run the deterministic stub
```

```
{"published": "…", "patient": "PT6580", "event_type": "VENTRICULAR_TACHYCARDIA",
 "confidence": 1.0, "criticality": "Critical", "mews": 6, …}
```

Watch it land:

```bash
docker compose -f infra/docker-compose.yml logs -f consumer
# [consume] received event daf16c2d… type=VENTRICULAR_TACHYCARDIA conf=1.00 patient=PT6580
# [consume] persisted MonitoredEvent … -> Neo4j graph
# [consume] archived report narrative -> Qdrant vector store
# [consume] dispatch=app: pushed inbox event … -> rmsai-inbox-h1 (kinds=[…])

# with `make demo-phone-up` (SIP), each gated event is queued, then dialled in severity order:
# [queued] PT6580/VENTRICULAR_TACHYCARDIA @ Unit1/Unit1-Bed01: call 1 in queue [app worklist pushed]
# [calls] dialling for PT6580/VENTRICULAR_TACHYCARDIA @ Unit1/Unit1-Bed01 (Critical, event daf16c2d…)
# [calls] skip PT6581/ATRIAL_FIBRILLATION @ Unit1/Unit1-Bed02: already acknowledged (event …)
```

Inspect the queue itself (event ids only; the job payloads carry pseudonyms, never names):

```bash
docker compose -f infra/docker-compose.yml exec redis redis-cli ZRANGE rmsai:calls:queue 0 -1 WITHSCORES
docker compose -f infra/docker-compose.yml exec redis redis-cli HLEN rmsai:calls:jobs   # queued + in progress
```

Then **use it**: the worklist in the open app fills live. Selecting a row scopes chat to that event
and, with the header's 🔊 **Voice on select** toggle on, speaks its report summary
(`INBOX_SPEAK_ON_SELECT`). In `app+call` mode the toggle starts muted (🔇), so the app doesn't
repeat what the phone call is saying; click it to turn the voice on. Ask questions by typing or by
voice. For the phone/WebRTC paths and the on-demand call, see
[End-to-end testing](TESTING.md#end-to-end-testing) §5–6.

Query the KB directly without the app:

```bash
$RMSAI cli.kb "which conditions are co-morbid with atrial fibrillation"   # hybrid: vector + graph
$RMSAI cli.text_chat                                                       # PIN-gated text console
```

### 5. Load documents into the KB

Protocols, SOPs, and checklists — PDF, markdown, or text — so questions are answered from *your*
documents with a citation instead of declined:

```bash
$RMSAI cli.kb_upload --file protocols/af_sop.pdf --dry-run   # preview the chunk plan
$RMSAI cli.kb_upload --file protocols/af_sop.pdf
$RMSAI cli.kb_upload --dir protocols/ --glob '*.pdf' --extract   # + graph entities

# a sample SOP ships with the repo, deliberately outside the auto-indexed corpus
$RMSAI cli.kb_upload --file docs/samples/critical_alarm_sop.md
```

Uploads are idempotent, copied into `KB_UPLOAD_DIR` so a rebuild re-indexes them, and PDF pages are
cited individually. Full detail + verification commands:
[§3a](TESTING.md#3a-upload-protocols--sops--checklists-to-the-kb).

### 6. Test the application

```bash
$RMSAI pytest -q                        # full suite (needs the stores up; graph-wiping tests skip unless RMSAI_ALLOW_LIVE_GRAPH_RESET=1)
$RMSAI pytest -q -m "not infra"         # offline only — no containers required
$RMSAI pytest -q tests/test_criticality.py -k vitals_driven   # one file / one test
make test                               # host equivalent (uv run pytest -q)
```

> **Known failures:** none offline (`-m "not infra"`: 1157 passed, checked 2026-10-14). The full
> suite (stores up) fails `test_orchestrator::test_multi_turn_state_persists`: with the hashing
> embedder the question does not pass the relevance gate. Treat a run as green when only that fails.

Then the end-to-end smoke test — generate, classify, publish, consume, alert, acknowledge:

```bash
$RMSAI cli.ingest --file data/inference/<f>.h5 --emit bus
docker compose -f infra/docker-compose.yml logs --tail=40 consumer
```

> These smoke tests exercise **plumbing, not accuracy**, so they take whatever `ECG_CHECKPOINTS`
> holds (or the stub, if it is unset). Note the input here is *simulator*-generated: `real_v2` is
> trained on real ECG and scores 10–26 % on synthetic data, so its labels on these files are
> meaningless even though the pipeline is working. To judge the model itself, feed it real
> recordings; to check a simulator file classifies plausibly, pass
> `--checkpoint external/ecgtranscnn/models/noise_robust/best_model.pt` explicitly.

Expect every event persisted. Gated events are dispatched with a reason: `ok`, `vitals_alert (…)`
(rhythm unconfirmed, vitals carry it; worklist only unless `outbound.call_vitals_alerts: true`) or `fp_override (…)` (confident NORMAL_SINUS, but vitals warrant
it; common on simulator data). The rest are skipped with a reason: `false_positive` (confident
NORMAL_SINUS), `low_confidence_arrhythmia (…)` or `below_threshold (…)`. A re-ingested event that was
already acknowledged/resolved is persisted but not re-alerted (`already_acknowledged`); run
`cli.events --reopen` to reset a demo. Per-subsystem harnesses are listed under
[Other CLI harnesses](TESTING.md#other-cli-harnesses-per-subsystem).

> ⚠️ **`cli.kb_eval` without `--no-seed` reseeds Neo4j: it refuses (exit 2) unless
> `RMSAI_ALLOW_LIVE_GRAPH_RESET=1`, and with it set it calls `reset_all()` on the LIVE graph** — that is
> why `cli.kb_dump --list` can return `[]`. The pytest tests that do the same
> (`test_graph_templates.py`, `test_orchestrator.py`, `test_outbound_flow.py`, …) are skipped unless
> `RMSAI_ALLOW_LIVE_GRAPH_RESET=1`. Recover by re-running step 3 then step 4.

### 7. Debug and inspect

**Dump everything stored for one event** — the graph node, the report file, and the vector chunks,
side by side. This is the first thing to reach for when an answer looks wrong:

```bash
$RMSAI cli.kb_dump --list                    # recent event ids to pick from
$RMSAI cli.kb_dump <event_id>
$RMSAI cli.kb_dump <event_id> --json         # raw {graph, vector, report_text}
```

```
=== EVENT daf16c2d-51fb-4cbe-a4ef-dea59502e56b ===
GRAPH (Neo4j)
  patient    : PT6580        bed : Unit1-Bed01
  event_type : VENTRICULAR_TACHYCARDIA | criticality Critical | status reported | FP False
  confidence : 0.9999  | MEWS risk High
  vitals     : HR 175.0, BP 175.0/93.0, SpO2 93.0, RR 24.0, Temp 99.0
  actions    : MEWS 6 (High) — escalate care
  plots      : ecg=data/plots/daf16c2d….png
  report     : report:daf16c2d…  (index_status indexed)
REPORT FILE (markdown) …
```

**Graph queries** — the operational templates, or a natural-language lookup that shows which
template it resolved to:

```bash
$RMSAI cli.graph template outstanding_action_items
$RMSAI cli.graph lookup "critical events in the last 24 hours"
#   -> {"mode": "template", "template": "critical_events_since", "rows": [...]}
```

**Event history: what is actually stored, and what would the agent say?**

```bash
$RMSAI cli.events --bed "bed 2" --status pending       # the rows (also: --patient, --type, --since, --json)
$RMSAI cli.events --ask "say all unacked events in bed 02"
# [events] query      event_history {"status": "pending", "bed": "Unit1-Bed02"}
# [events] rows       2
# [events] answer     2 pending (unacknowledged) events on bed Unit1-Bed02 for 1 patient: …
```

**Why did a question take that path?** Regex template vs LLM router vs document retrieval:

```bash
$RMSAI cli.kb_route "what were the vitals at the event"
# [route] regex -> (no match)
# [route] looks operational? True (the LLM router is eligible)
# [route] pass --llm to try the LLM router
```

**Vector store** — what's actually indexed, and does the right passage win:

```bash
$RMSAI cli.kb_vector --embedder bge retrieve "escalation checklist for a critical alarm" -k 3
curl -s -X POST http://localhost:6333/collections/rmsai_docs/points/scroll \
  -H 'Content-Type: application/json' -d '{"limit":500,"with_payload":true}' \
| python3 -c "import sys,json,collections; c=collections.Counter(p['payload'].get('doc_id') for p in json.load(sys.stdin)['result']['points']); [print(f'{v:4d}  {k}') for k,v in sorted(c.items())]"
```

**In-app chat / push-to-talk not responding?** Probe the room without a browser — this splits a
worker fault from a browser fault (stale `app.js`, handler never fired):

```bash
$RMSAI cli.inbox_probe --ptt
$RMSAI cli.inbox_probe --select <event_id> --say "what were the vitals at the event?"
$RMSAI cli.dispatch --all-inbox       # re-wire live inbox rooms that lost their agent
```

**Logs, audit trail, and GUIs:**

```bash
make docker-logs                                    # all three app services, follow
docker compose -f infra/docker-compose.yml logs -f voice-worker
tail -f data/audit.jsonl                            # {ts, actor, action, subject, outcome}
```

Neo4j Browser `http://localhost:7474` · Qdrant dashboard `http://localhost:6333/dashboard` ·
companion app `http://localhost:8080/`.

**Common failures and what they actually mean:**

| Symptom | Cause |
|---|---|
| `Error 111 connecting to localhost:6379` | the backing stores aren't running — `make stores-check`, then `make stores-up`. Running the CLIs under `uv` does not make them self-contained |
| `[errno 98] address already in use` on 8080/8081 | a host-run gateway/worker is still up — host networking shares the port space |
| worker retry-loops on `:7880` | the LiveKit **server** isn't running, not a worker bug |
| `collection … has vector dim 384, but embedder … 256` | `--embedder` doesn't match what built the collection |
| `cli.kb_dump --list` returns `[]` | the graph was wiped (`cli.kb_eval` or pytest, run with `RMSAI_ALLOW_LIVE_GRAPH_RESET=1`) — redo steps 3–4 |
| worklist empty | nothing is pending: every event was acknowledged or withheld (check the **Events** tab or `cli.events --status pending`). Also: `DISPATCH_MODE` without `app`, or `[retry]`/`[dead] … refusing to publish non-pseudonym patient ref` (the message ends up in `rmsai.events.dead`; curate real ECG with `cli.real_samples`) |
| worklist empty, chat silent | worker not dispatched into the room → `cli.dispatch --all-inbox`; or a cached `app.js` (check the on-screen build tag) |
| selecting a row is silent | the header shows 🔇 **Voice muted** (the default in `app+call` mode); click it. The worker logs `selected event … (app voice muted: not spoken)` |
| the worklist shows events whose call hasn't come yet | expected with `--transport sip`: calls queue and go out one at a time, most severe first. `[queued] … call N in queue` in the consumer log gives the position |
| answer declines on an in-corpus question | relevance gate — the log names both signals and both thresholds |

### 8. Performance

**Retrieval quality + cost + latency**, vector vs hybrid over the gold question set — the go/no-go
on hybrid. Reports correctness, citation grounding, context-token cost, and per-mode latency:

```bash
$RMSAI cli.kb_eval                     # refuses unless RMSAI_ALLOW_LIVE_GRAPH_RESET=1 (then resets the live Neo4j); --no-seed is safe
$RMSAI cli.kb_eval --json
```

**Speech latency per leg** — TTS → STT round-trip with no audio hardware, the way to compare a
self-hosted backend against a cloud one:

```bash
$RMSAI cli.speech_check                                    # whisper + piper (self-hosted)
$RMSAI cli.speech_check --tts elevenlabs --stt elevenlabs  # cloud (synthetic text only)
```

**Container resource use:**

```bash
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'
# infra-consumer-1      0.10%   262.5MiB / 15.35GiB
# infra-voice-worker-1  1.25%    72.8MiB / 15.35GiB   (idle; a live job forks a ~600MB child)
# infra-gateway-1       0.27%    35.0MiB / 15.35GiB
```

The voice worker forks a **job process per room**; livekit-agents logs a memory warning above
500 MB (advisory) and kills a job process that stops answering health pings — which is what a slow
cold start inside a contended container looks like. Per-turn spans (`common/tracing.py`) are
recorded on every `TurnResult.trace` for step-level latency.

Knobs that move the needle: `WHISPER_MODEL` (`tiny.en` ≫ faster than `base.en`), `EMBEDDER`
(`hashing` skips the BGE load entirely), `LLM_PROVIDER=echo` (no model call at all),
`KB_LLM_ROUTER=false` (default — it adds a model call in front of a Cypher lookup, ~6s on
llama3.2:3b), and `EPISODIC_RECALL=false` (default).

### 9. Shut down

```bash
make docker-down          # stop + remove containers; named volumes survive
```

Data survives a `down`: Neo4j (`neo4j_data`), Qdrant (`qdrant_data`), the model cache (`hf_cache`),
and everything under `./data` (reports, plots, uploads, audit log) are volumes or bind mounts.
Redis is in-memory by design (`--save ""`), so the **bus backlog, working memory and the call queue
do not survive**: a call still queued at `down` is never placed (the event itself stays in Neo4j and
on the worklist as pending). The call queue does survive a consumer-only restart.

```bash
docker compose -f infra/docker-compose.yml down -v      # ⚠️ ALSO deletes graph + vectors + model cache
docker compose -f infra/docker-compose.yml stop consumer   # stop one service
docker compose -f infra/docker-compose.yml restart voice-worker
```

After `down -v` you are back to step 3 — re-initialize the KB, then re-ingest.

## Operational query matrix (the verification target)

Every row is answerable through tested, read-only, parameterized Cypher templates (no free
text-to-Cypher; with `KB_LLM_ROUTER=true`, operational questions the rules miss go to the LLM router,
which picks a template). **Every question the agent understands, with examples, is listed in
[`docs/project/queries.md`](docs/project/queries.md)**; the matrix below is the original spec set:

| # | Query | Store |
|---|-------|-------|
| 1 | Critical events last 24h by patient/bed/unit (criticality High+, i.e. call-worthy) | graph |
| 2 | Positive (non-FP) events last *x* min | graph |
| 3 | Event status on a bed (ts, event, FP?, actual condition) | graph |
| 4 | Event analysis report for a patient/bed/unit | graph + vector |
| 5 | Vitals at the time of a specific event | graph (inline snapshot) |
| 6 | Outstanding action items across patients | graph |
| 7 | Care protocol for a bed's last event | hybrid |
| 8 | Patterns: age/gender/co-morbidity/symptom → event type | graph (analytics) |
| 9 | ECG strips for the last event (any type, a named type, or *this patient's*) | graph → PNG artifact |
| 10 | HR & BP trend for last Tachycardia event; HR-history trend for *this patient's* event | graph → plot ref / HR series |
| 11 | Event history: events by patient / bed / type / status (pending, acknowledged, withheld), optional last *N* h | graph (`list_events`, the Events tab's query) |

## Example questions the system answers

These are the natural-language prompts a clinician can ask (over text or an authenticated call) that
the data model is built to serve. Each maps to a tested query template (`T*` above) or the hybrid
retriever:

| Question | Maps to |
|----------|---------|
| "List all patients, bed number, unit/ward who had **critical events in the last 24 hours**." | T1 (graph) |
| "List **positive patient events** reported in the **last *x* minutes**." | T2 (graph, non-FP) |
| "What is the **status of events reported on Bed xx** — timestamp, reported event, false positive?, actual condition?" | T3 (graph) |
| "Get the **event analysis report** for patient / Bed xx in Unit/Ward." | T4 (graph → vector report content) |
| "What were the **vitals at the time of the specific event** for the patient in Bed xx?" | T5 (graph, inline vitals snapshot) |
| "Provide an **action-item list** of all outstanding actions for patients." | T6 (graph) |
| "What is the **treatment / care protocol** for Bedside x's last reported event?" | T7 (hybrid: graph condition + protocol, narrative from vector) |
| "From the data, do you see any **pattern of age / gender / co-morbidities / symptoms** leading to a specific event?" | T8 (graph analytics — framed as correlation, not causation) |
| "Show me the **ECG strips** for the patient with the last reported (AFib) event." | T9 (graph → PNG path; producer renders the lead, voice says "an ECG strip is available", companion app shows it) |
| "Show me the **HR and BP trend** for the patient with the last reported Tachycardia event." | T10 (graph → vitals/trend plot ref) |
| "**How was HR trending** at the time of this patient's event?" | HR-history series persisted on the event (POC: HR; RR/SpO2/BP later) — answered as "HR trended from X to Y (rising)…" |
| "What are the **other critical / all events for *this patient***?" | session-patient-scoped (outbound call) — filtered to the bound patient, not all |
| "Show **all patients who have a specific event** (e.g. all AFib)." | graph traversal: `MonitoredEvent {event_type}` → `Patient` (+ optional bed/unit) |
| "Say all **unacknowledged events in bed 2**" · "**acknowledged** events for this patient" · "all **VT events** in the last 6 hours" · "**withheld** events" | T11 event history: the same rows as the app's Events tab and `cli.events` |

**Verifiable event listings.** Every event-history answer (T11) states the filter it used, e.g. "2
pending (unacknowledged) events on bed Unit1-Bed02 for 1 patient: …". A listing across patients also says how many patients ("10 pending (unacknowledged)
events for 9 patients: …"). A typed answer lists each event's time,
confidence, criticality, status and short id. *Pending* has one meaning everywhere: **alerted**
and not acknowledged or resolved. An event that was never alerted is *withheld*.

To check a spoken list:
- open the **Events** tab with the same filter (in the app chat, the "📋 Open in Events tab"
  button under the answer does it), or run `cli.events` with the same flags;
- run `cli.events --ask "<question>"` to see the exact query, its parameters, the rows and the
  answer the agent gives.

A listing whose target can't be resolved ("events for Mr Smith") is declined with a hint rather
than guessed.

> Operational items (1–6, 9, 10) are judged by **exact-row match**; hybrid/relationship items (7, 8)
> by **groundedness + citations**. Out-of-corpus / unknown-bed / unknown-patient questions are
> expected to **decline**, never fabricate.

Event names in the event-scoped questions (9, 10, "all patients with …") are **parameterized**:
substitute any of the 16 vocabulary classes — "AFib", "v-tach", "ST elevation", "mobitz 2", "SVT",
… — and the same template runs with a different `event_type` (NL→class resolved by
`event_type_from_text`). Resolution covers all 16; *detection* is limited to the loaded checkpoint's
head, so with `real_v2` a query for "ST elevation" or "mobitz 2" is well-formed but will never match
a stored event — that model cannot predict those three classes at all.

The matrix is wired identically for **voice and text** (the voice handler routes through the same
`match_intent`). Spoken queries are normalized first (`kb/graph/spoken.py`) so STT phrasing resolves
like typed: spelled-out acronyms collapse ("A V Block" → "AV Block", "S V T" → "SVT"), number-word
bed labels rebuild ("bed unit one bed oh one" → "Unit1-Bed01"), and spoken counts become digits
("twenty four hours" → 24). So bed/event-type/time-scoped questions work spoken, not just typed.

**Answer style.** Operational (template-matched) questions are answered **deterministically** from the
graph rows — crisp, exact, no LLM in the loop (so a small local model can't pad or garble them), and
with no conversation history or recalled context in the path. Only free-text/hybrid questions go
through the LLM, under a tight instruction to lead with the answer and drop preamble/disclaimers; that
path includes the live conversation history (for follow-ups) but recalled cross-session "past
interactions" only when `EPISODIC_RECALL=true` (off by default). Patient pseudonyms (`PT####`),
clinical terms ("SVT"), and bed/unit labels are preserved through de-identification (presidio NER
would otherwise scrub them as PII).

## Managing & inspecting the KB

Two stores hold an event: the **graph** (Neo4j) is the structured source of truth (the
`MonitoredEvent` + vitals snapshot + links + a `Report` node whose `uri` points at the materialized
`data/reports/<id>.md`); the **vector** store (Qdrant `rmsai_docs`) holds the report *narrative*
(`doc_id=report:<id>`, served by report lookups and kept out of free-text answers) plus the
clinical-protocol corpus.

**Dump what's stored for one event** (graph node + vector chunks + report file, side by side):
```bash
uv run python -m cli.kb_dump --list             # recent event ids
uv run python -m cli.kb_dump <event_id>          # graph + report file + vector chunks
uv run python -m cli.kb_dump <event_id> --json   # raw {graph, vector, report_text}
```
Ad-hoc graph reads use `cli.graph` (templates or read-only Cypher); GUIs: Neo4j Browser
`http://localhost:7474`, Qdrant dashboard `http://localhost:6333/dashboard`.

**Indexing is append-by-default — re-indexing docs no longer wipes event narratives.**
`cli.kb_vector index` replaces each indexed document's chunks (sections removed from a document stop
being searchable) and leaves every other document, including event-report narratives, alone; pass
`--reset` only for a clean rebuild. The
event-writing/serving paths (`cli.consume`, `cli.outbound`, and `build_orchestrator` — which runs on
every text chat **and** every voice call) all **append**, so they preserve the report narratives that
`consume` archives. Append requires a **matching embedder dimension**: the collection is built with
one embedder (hashing=256 / BGE=384), so those paths default `--embedder` to `EMBEDDER` from `.env`;
re-index with the same embedder, or `--reset` to rebuild — a clear error fires on mismatch.

```bash
uv run python -m cli.kb_vector index --dir docs              # append (preserves event reports)
uv run python -m cli.kb_vector index --dir docs --reset      # full rebuild (wipes the collection)
```

**The corpus is `docs/*.md` plus the managed upload folder.** `chunk_dir` globs `docs/` **non-
recursively for `*.md`**, so every markdown file dropped there (except `README.md`) becomes retrievable clinical
evidence — project/engineering docs belong in `docs/project/` and upload test fixtures in
`docs/samples/`, both of which the glob excludes. Non-markdown files are never picked up by the
glob: a PDF must go through `cli.kb_upload`. Documents added that way are copied to
`KB_UPLOAD_DIR` (`data/kb_uploads/`, gitignored) and re-indexed by every rebuild, so they are part
of the corpus rather than a one-off write. See [§3a](TESTING.md#3a-upload-protocols--sops--checklists-to-the-kb)
for the upload + verification commands.

**How a question is answered** — three paths, in order:

| path | when | answer |
|---|---|---|
| graph template | `match_intent` regex hits | exact patient data from Cypher, no LLM |
| LLM-routed template | regex misses, question looks operational, `KB_LLM_ROUTER=true` | same, template chosen by the model from a fixed menu |
| hybrid retrieval | everything else | cited passages (vector) + relationships (graph), answered by the LLM |

`uv run python -m cli.kb_route --llm "<question>"` shows which path a question takes and why.

**Relevance gate.** A hybrid answer is only given when retrieval vouches for itself: a graph
relationship, semantic similarity ≥ `KB_MIN_RELEVANCE` (0.60), **or** word overlap ≥ 0.18. The two
signals fail in opposite directions — semantic similarity survives rewording (calibrated on this
corpus: on-topic 0.61–0.83, off-topic 0.35–0.54, but only under `EMBEDDER=bge`; hashing scores the
two alike), while word overlap is what carries the hashing embedder. When neither vouches, the turn
declines and logs **why**, naming both numbers, both thresholds, and the top passage. Even when one
vouches, a question whose topic words appear in no retrieved passage is declined unless similarity is
at least 0.70 (`[orchestrator] declined: no retrieved passage mentions …`):

```
[orchestrator] declined: 4 passage(s) retrieved, but neither relevance signal vouched for them:
semantic similarity 0.17 < 0.60 AND word overlap 0.00 < 0.18 (top: vt_vf.md#Post-resuscitation)…
```

> ⚠️ **Destructive ops to know about.** `GraphDriver.reset_all()` refuses to wipe Neo4j unless
> `RMSAI_ALLOW_LIVE_GRAPH_RESET=1` (`kb/graph/driver.py`), and the tests that call it are skipped
> without it (`conftest.py`). Set it only against a graph you can lose: with it set, `cli.kb_eval`
> (without `--no-seed`) resets the live Neo4j; without it, kb_eval stops with an error.
> `cli.kb_vector index --reset` rebuilds the Qdrant collection. After any such wipe, re-run the `ingest → consume` flow to
> repopulate events (and their report narratives).

## Model performance & traceability

For every event the relay records, and shows in the logs, the CLI and the companion app:
- **where the data came from:** e.g. `MIT-BIH 207 @221697 (test split)`, `simulator`, or a device id;
- **which model said what:** e.g. `ecg_transconv:v2:096bdfbafa04x5` at 63 %;
- **whether that was right**, when a ground truth exists;
- **why the clinician is (or isn't) seeing it.**

**Outcome per event** (`inference/metrics.py`; positive = any arrhythmia, i.e. not `NORMAL_SINUS`):

| Outcome | Meaning |
|---|---|
| **TP** | arrhythmia called arrhythmia; ✓ if the class is exact |
| **TP_WRONG_CLASS** | arrhythmia called a *different* arrhythmia: the alarm was right, the class wasn't |
| **FP** / **FN** / **TN** | false alarm / missed arrhythmia / normal called normal |
| **UNSCORABLE** | no ground truth, or a label the model can't output (e.g. `OTHER`, or `ST_ELEVATION` for the 13-class `real_v2`) |

The **alert level** scores what was *dispatched* after the criticality gate (correct, missed, false,
silent-correct). It's the clinically meaningful one, because vitals can raise an alert the rhythm
doesn't support. Rates (sensitivity, specificity, PPV, NPV, exact-class accuracy, per-class
precision/recall, confusion matrix) come with **Wilson 95 % intervals** and their raw *k/n*: a demo
set of a few events gives wide intervals, and the view says so.

**Why an event is shown** (`orchestrator/explain.py`) restates the relay's own decision functions,
so it can't drift from the behaviour:
- the rhythm's confidence against the threshold (asserted / unconfirmed / normal);
- MEWS against its threshold, with the contributing vitals;
- **every** deteriorating vital, with direction and p-value;
- what raised the criticality;
- the gate decision.

It ends in one headline, e.g. *"Shown because the vitals warrant it, not the rhythm: systolic BP
falling (p<0.001), HR rising (p<0.001), SpO₂ falling (p=0.0091). Rhythm reads normal (58%). High."*

**Where it shows:**

| Surface | What |
|---|---|
| `cli.ingest --explain --metrics` | `outcome` + `why` per event; the full breakdown on stderr; `--trend-samples` adds the readings behind each trend |
| consumer log (`make docker-logs`) | one `[event]` line per event (source · prediction vs truth → outcome · criticality · alert + delivery · why; with `--trend-samples` in `CONSUME_ARGS`, also the readings behind each trend); a `[perf]` summary every `--perf-every` labelled events (default 10) and on exit |
| `cli.model_perf` | the summary on request, from the graph: `--since 30m\|24h\|7d`, `--model`, `--dataset`, `--events`, `--json` |
| companion app | worklist rows show the why line, a source badge (demo data only) and the outcome; selecting a row shows the full explanation + data source, and each "rising/falling" vital expands to the readings it was computed from (with a mini chart); the **Events** tab lists every stored event (filters: status, patient, bed, type, criticality, time), and its **Labelled only** mode adds the tiles, confusion matrix and per-class table over the listed events, live-refreshing as events arrive |
| graph | each `MonitoredEvent` stores `source_*`, `model_id`, `alert_gate` / `alert_reason_code`, `why` (+ `why_json`), `eval_outcome`, `delivered_app/call/sms`, `processed_at` |

```bash
$RMSAI cli.model_perf --events            # all labelled events, one block per model
$RMSAI cli.model_perf --since 1h --dataset mitbih
```

**Notes:**
- **Ground truth is evaluation-only.** It's stored and shown, but the event's clinical condition link
  is always the *prediction*, so graph answers never use the answer key.
- **Data source:** comes from the file. ecg_sigma recordings carry dataset, record, subject, the
  sample offset in the original recording, the labelling method/purity, and the package/split.
  Simulator and device files carry only their device id.
- **Time filter:** `cli.model_perf --since` uses **when the relay processed** an event
  (`processed_at`), not the recording time, which for public datasets can be decades old. The app's
  Events tab and `cli.events --since` filter on the **event time** (the HDF5 time by default; with
  `INGEST_TIME_ANCHOR=now` or `cli.ingest --time-anchor now`, each patient's events are shifted so
  the latest is the ingest time).
- **Model mixing:** metrics are grouped **per model**; a summary mixing two models describes neither.
- **Production data has no ground truth.** The performance view is for curated, labelled sets
  (`cli.real_samples`); on unlabelled data it says so.
- **Display precision:** vitals are shown as integers, and temperature to one decimal, everywhere
  people read or hear them (chat/voice answers, reports, the explanation, the app). Stored values stay
  raw, since the trend tests and MEWS use them.
- **Trend evidence:** "HR rising (p<0.001)" comes from a Mann-Kendall test on the vital's history in
  the event window (typically 25–30 readings). The app and `--trend-samples` show exactly those
  readings.
- **Synthetic vitals:** for ecg_sigma data the vitals are synthetic (SpO₂/BP condition-keyed), so the
  vitals-driven parts of the alert-level metrics partly echo the label.
