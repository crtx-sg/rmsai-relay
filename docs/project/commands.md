# Commands reference

What the operational commands do, what they need, and what they touch. Add an entry whenever a
command needs more explanation than its `--help` gives.

> Keep this file in `docs/project/`. Any `*.md` placed directly in `docs/` gets indexed as
> clinical KB evidence (see `docs/README.md`).

---

## Real-ECG samples (`cli.real_samples`)

### `docker compose -f infra/docker-compose.yml run --rm real-samples list`

**Purpose:** a read-only inventory. It shows how many held-out **real** ECG events per arrhythmia
class are available in an `ecg_sigma` training package. Run it before `pick`, so you know what a
`--pick LABEL:N` can draw on.

**How the command breaks down:**

| Part | Meaning |
|---|---|
| `-f infra/docker-compose.yml` | Use the infra compose file. The Compose project dir becomes `infra/`, so the repo-root `.env` is **not** read for `${...}` substitution; export `ECGPKG_HOST_DIR` / `ECGPKG_NAME` in the shell if you need them. |
| `run --rm` | One-shot container that is removed on exit. Nothing long-running starts. |
| `real-samples` | Service in the `tools` profile (`run` starts it regardless of profile). It uses the same app image as `tools`, with entrypoint `python -m cli.real_samples`. |
| `list` | The subcommand (also the service's default `command`). |

**Mounts / env set up by the service:**

- Repo bind-mounted at `/app`, so the container sees your working tree.
- `${ECGPKG_HOST_DIR:-../../ecg_sigma/packages}` (relative to `infra/`, so it resolves to
  `../ecg_sigma/packages` next to the repo) is mounted **read-only** at `/ecgpkg`.
  `create_host_path: false`: if the dir is missing, the command fails loudly. It does not mount an
  empty dir.
- `ECGPKG_DIR=/ecgpkg/${ECGPKG_NAME:-ecg_pkg_v2}` is the package `cli.real_samples` reads by default.

**What it does:**

1. Loads `package.json` + `manifest.csv` from the package. No HDF5 is opened.
   Fails with exit 2 if it isn't an `ecgpkg`.
2. Counts the manifest rows in the chosen split (default `test`) per label, broken down by source
   dataset.
3. Prints a header line, then one line per class (the counts below are examples):

   ```
   package <version> (manifest <sha12>), split=test
     VENTRICULAR_TACHYCARDIA         123  (mitbih:40, vfdb:83)
     ATRIAL_FIBRILLATION              ...
     NORMAL_SINUS                     ...
   ```

   `-` means no events for that class in the split.

**Side effects:** none. It writes no files, loads no model and touches no stores (Neo4j, Qdrant,
Redis).

**Variants:**

```bash
# another split (train/val/test)
docker compose -f infra/docker-compose.yml run --rm real-samples --split val list

# a different package / location
ECGPKG_HOST_DIR=/path/to/packages ECGPKG_NAME=ecg_pkg_v3 \
  docker compose -f infra/docker-compose.yml run --rm real-samples list

# host equivalent (no Docker)
uv run python -m cli.real_samples list
```

Note: `--split` and `--package` are top-level options, so they go **before** the subcommand.

### Next step: `pick` → `ingest`

```bash
docker compose -f infra/docker-compose.yml run --rm real-samples \
  pick --pick VENTRICULAR_TACHYCARDIA:3,ATRIAL_FIBRILLATION:2,NORMAL_SINUS:2
docker compose -f infra/docker-compose.yml run --rm tools \
  python -m cli.ingest --dir data/real --emit bus
```

- `pick` writes reader-ready HDF5 into `data/real/` and **removes earlier pick files there first**.
  It prints one JSON line per event (label, `PT9xxxxx` pseudonym, dataset, event key) and then a
  summary line.
- `pick` refuses (exit 2) if the package's manifest sha doesn't match the checkpoint's training
  package, because the `test` split is only held out for the model trained on that package.
  Override with `--allow-mismatch`.
- `--split train|val` prints a warning: the model has already seen those events, so results are not
  a performance estimate.
- With no `ECG_CHECKPOINTS` / `--checkpoint`, `pick` cannot check the manifest, and `cli.ingest`
  then runs the ECG **stub**.
- Other options: `--dataset mitbih vfdb`, `--seed N` (repeat an earlier draw), `--out data/real`, `--checkpoint <pt…>` (checkpoint(s) to check the package manifest against; default `ECG_CHECKPOINTS`).

### `uv run python -m cli.real_samples pick --pick VENTRICULAR_TACHYCARDIA:3,VENTRICULAR_FIBRILLATION:2,ATRIAL_FIBRILLATION:3,PVC:2,LBBB:1,PAC:2,NORMAL_SINUS:2`

**Purpose:** builds a 15-event demo/eval set of real ECG from the package's held-out `test` split
(3 VT, 2 VF, 3 AF, 2 PVC, 1 LBBB, 2 PAC, 2 normal sinus). It writes the set as reader-ready HDF5 in
`data/real/`, ready for `cli.ingest --dir data/real`.

This runs on the host, so the package is `$ECGPKG_DIR` if set, otherwise
`../ecg_sigma/packages/ecg_pkg_v2`.

**Steps, in order:**

1. **Parse `--pick`:** labels are upper-cased and must be one of the 16 `ecgtranscnn` classes.
   Unknown labels or counts below 1 cause exit 2. Repeated labels have their counts added.
2. **Manifest check:** reads the training-package sha stored in the first `ECG_CHECKPOINTS`
   checkpoint (`.env` points to `real_v2/fold0..4.pt`). If it doesn't match the package's
   `manifest_sha256`, the command stops with exit 2 unless `--allow-mismatch` is passed.
   Without this check, "held out" would mean nothing.
3. **Select:** for each label, gets the `test`-split candidates, sorts them by (file, event key)
   and samples N with `random.Random(--seed)`. Without `--seed` each run draws a new random seed and prints it
   in the summary line; the same package, arguments and seed always give the same events. If a label has too few candidates, the command takes what exists and prints a
   shortfall instead of failing.
4. **Pseudonymise:** each source `subject_id` maps to a stable `PT9#####` pseudonym (sha256-based).
   Two recordings of one subject become one patient. A collision between distinct subjects aborts
   the run before anything is written; re-run with another `--seed`.
5. **Clear `data/real/`:** deletes earlier pick outputs, identified by the `ecgpkg_version`
   metadata stamp. Any other `.h5` files are kept, with a warning, and `cli.ingest --dir` will
   read them too.
6. **Write:** one output file per source record, containing `/metadata` plus only the selected
   event groups. `patient_id` is replaced with the pseudonym, and the original is kept in
   `source_patient_id`. Each event's `condition` attribute is set to the **manifest label**, which
   becomes the ground truth `cli.ingest` scores against. The original annotation is kept in
   `source_condition`. The command refuses to write inside the package, because the package is
   checksummed.
7. **Output:** one JSON line per event (`label`, `patient`, `dataset`, `subject`, `event_key`,
   `file`), then a summary line (`events`, `files`, `package`, `split`, `seed`). Exit 0 if any file
   was written.

**Side effects:** only `data/real/` changes. No stores, no bus and no model inference are involved.
The model only runs at the next step, `cli.ingest`.

**Availability:** in `ecg_pkg_v2`'s `test` split (as of 2026-10-12), every requested label has well
above the counts asked for (the smallest is LBBB with 81), so expect no shortfalls.

---

## Demo stack

### `make demo-up` vs `make demo-phone-up`

| Target | Consumer transport | Phone worker (`rmsai-agent-phone`) | Reportable events… |
|---|---|---|---|
| `make demo-up` | `--transport webrtc` (compose default) | not started | stage a WebRTC room and print a join token. **No phone rings.** The log still says `call answered`, because a simulated caller "answers". |
| `make demo-phone-up` | `--transport sip --notifier signalwire` | started (`telephony` profile) | dial `OUTBOUND_CALL_NUMBER` through LiveKit Cloud + SignalWire, one call at a time; an unanswered call is texted from the same number (a pickup with no PIN is unanswered: retried, then texted; a PIN lockout is neither). |

Both targets finish with `infra/demo-check.sh`. Running `make demo-up` after `demo-phone-up`
recreates the consumer back on WebRTC. Vitals-driven alerts (rhythm below the confidence bar)
only ring if `OUTBOUND_CALL_VITALS_ALERTS=true`.

Phone smoke test, independent of any event (rings your mobile and plays the PIN prompt; it redials on
no answer but never texts, so test the SMS path by publishing an event instead):

```bash
docker compose -f infra/docker-compose.yml run --rm tools python -m cli.call --caller livekit
make phone-logs   # expect "caller answered" then "speaking greeting"
```

In SIP mode the consumer pushes **every** gated event, plus vitals alerts, to the app as soon as it
arrives. Calls go to a Redis call queue (`rmsai:calls:*`) and are placed one at a time across the whole
system (a single call slot, `rmsai:calls:active`, with a 60 s lease, so two consumers never ring
twice), highest
criticality first and first-come within a level. A queued call is skipped if the event was
acknowledged (in the app) or resolved while it waited. Calls still waiting survive a consumer
restart. Console lines: `[queued] … call N in queue`, `[calls] dialling …`/`skip …`,
`[call] … -> answered`, `[consume] call in … ended with no PIN entered: counted as missed`,
`[calls] call for … dropped on a network reconnect; ringing again`.

### `make phone-transcript`

Follows the phone worker and shows only the conversation: `stt: ->` (raw transcript), `heard:`,
`reply:` (the KB-grounded answer, logged before TTS speaks it), the greeting, `[voice] PIN accepted`
and `[voice] alert text:` (the event alert spoken after the PIN). It shows new lines only, so start
it before the call. Ctrl-C stops it.

---

## Event history (`cli.events`)

Lists every persisted event from the graph (alerted or not), newest first, with optional filters.
Read-only except `--reopen`. It uses the same query and the same **pending** definition (alerted AND not
acknowledged/resolved) as the companion app and the voice agent, so it's the way to check a spoken
answer.

```bash
uv run python -m cli.events                                  # everything
uv run python -m cli.events --bed "bed 2" --status pending   # Bed02 in any unit; or Unit1-Bed02
uv run python -m cli.events --patient PT998224 --since 7d
uv run python -m cli.events --type VENTRICULAR_TACHYCARDIA --labelled
uv run python -m cli.events --status withheld                # not alerted (low confidence, FP)
uv run python -m cli.events --min-criticality Critical --json

# how a voice/chat question resolves: normalized text, graph query + params, rows, the answer
uv run python -m cli.events --ask "say all unacked events in bed 02"
uv run python -m cli.events --ask "unacknowledged events for this patient" --as-patient PT998224 --spoken

# demo reset (writes): set acknowledged/resolved events back to reported, so a re-ingest alerts again
uv run python -m cli.events --reopen
uv run python -m cli.events --reopen <event-id> <event-id>
```

Voice and chat event listings ("unacknowledged events on bed 2", "acknowledged events for this
patient", "all VF events", "withheld events") run this same query. Every answer states its filter
("2 pending (unacknowledged) events on bed Unit1-Bed02 for 1 patient: …"), and text answers give each event's
short id. A listing whose target can't be resolved ("events for Mr Smith") is declined with a
hint, not guessed. The worker log shows `[orchestrator] graph query event_history {…} -> N row(s)`,
and `make phone-transcript` includes that line.

`--status`: `any` (default), `pending`, `acknowledged`, `resolved`, `withheld`. Other filters:
`--unit Unit1`, `--model <id>`, `--dataset mitbih`, `--limit N`. `--since` filters on
event time. Each row shows time (UTC), patient, bed, type, confidence, criticality, status (or the
reason it was withheld), whether it was a vitals alert, what was delivered (app/call/SMS),
truth → outcome for labelled events, and the event id.

### Companion app: Events tab and worklist backfill

- **Events tab** (replaces "Model performance"): every stored event, with filters for status
  (pending / acknowledged / resolved / withheld), patient, bed, type, minimum criticality and time.
  Backed by `POST /events`, the same query as `cli.events` and the voice/chat listings. Each row
  shows a short event id, the alert decision, what was delivered, and truth/outcome for labelled
  events. Pending rows can be acknowledged from the tab. Click a row for the detail panel.
- **Labelled only** adds the per-model performance summary (confusion matrix, sensitivity, PPV …)
  over the listed events, plus the dataset and model filters. The time filter is on *event* time
  (the old tab used processing time).
- **Worklist backfill:** on login (and on page reload) the worklist loads every pending event, not
  only the ones pushed since you connected.
- **Chat → Events link:** when a chat answer lists events, a "📋 Open in Events tab: …" button
  appears under it and opens the tab with the same filter.
- **Voice on select:** the header toggle (🔊 Voice on select / 🔇 Voice muted) decides whether
  selecting a row speaks its report. It's muted by default in `app+call` mode, so a clinician on
  the phone alert doesn't hear the event twice. The choice is remembered per dispatch mode in the
  browser. The worker log shows `selected event … (app voice muted: not spoken)` for a muted
  selection.

## Knowledge graph (Neo4j)

The graph's node and edge types are described in ARCHITECTURE.md's **Knowledge-base data model**
section. Every query below is read-only, except `cli.graph chain`, which rebuilds edges.

### Neo4j Browser (visual tree)

Open **http://localhost:7474** (user `neo4j`, password `rmsai_dev_pw`), paste a query, and
double-click any node in the result to expand its neighbours.

```cypher
// whole schema: one node per label, one edge per relationship type
CALL db.schema.visualization()

// one patient's tree, 2 hops (events → report/actions/condition, bed → unit, history)
MATCH path = (p:Patient {id: 'PT998224'})-[*1..2]->(n)
RETURN path

// a patient's event history in time order (FOLLOWED_BY chain)
MATCH path = (:Patient {id: 'PT998224'})-[:HAD_EVENT]->(first)-[:FOLLOWED_BY*0..]->(e)
WHERE NOT ()-[:FOLLOWED_BY]->(first)
RETURN path

// one event and everything around it
MATCH path = (e:MonitoredEvent {id: 'feb5d443-b28b-5644-9857-049bed4e3d43'})-[*1..2]-(n)
RETURN path

// clinical knowledge: guidelines → conditions / treatments
MATCH path = (g:Guideline)-[:APPLIES_TO|RECOMMENDS]->(n)
RETURN path LIMIT 100
```

### Terminal (`cypher-shell`)

```bash
C="docker exec infra-neo4j-1 cypher-shell -u neo4j -p rmsai_dev_pw"

# schema as text: from-label, relationship, to-label, count
$C "MATCH (a)-[r]->(b) RETURN labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to, count(*) AS n ORDER BY from, rel"

# one patient's tree, 2 levels
$C "MATCH (p:Patient {id:'PT998224'})-[r1]->(n1) OPTIONAL MATCH (n1)-[r2]->(n2)
    RETURN type(r1) AS rel, labels(n1)[0] AS node, coalesce(n1.id, n1.label, n1.name) AS id,
           type(r2) AS rel2, labels(n2)[0] AS child, coalesce(n2.id, n2.label, n2.name) AS child_id"

# every patient's event chain, with the gap between events
$C "MATCH (p:Patient)-[:HAD_EVENT]->(a)-[f:FOLLOWED_BY]->(b)
    RETURN p.id AS patient, a.event_type AS first, b.event_type AS next, round(f.gap_s, 1) AS gap_s"

# node counts by label
$C "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"
```

The password is the dev default (`NEO4J_PASSWORD`); use yours if you changed it.

### Project CLIs

```bash
uv run python -m cli.kb_dump --list              # recent event ids
uv run python -m cli.kb_dump <event_id>          # one event: graph node + report file + vector chunks
uv run python -m cli.events --patient PT998224   # the patient's events, as the app shows them
uv run python -m cli.graph template <name>       # any operational query template
uv run python -m cli.graph chain                 # rebuild the FOLLOWED_BY chains (backfill; writes)
```

### What the tree looks like

```
Patient ─HAD_EVENT→ MonitoredEvent ─FOLLOWED_BY→ (the patient's next event)
   │                    ├─HAS_REPORT→ Report
   │                    ├─HAS_ACTION→ ActionItem
   │                    ├─OF_CONDITION→ Condition
   │                    └─AT_BED→ Bed ─IN_UNIT→ Unit
   ├─ASSIGNED_TO→ Bed
   ├─HAS_DIAGNOSIS→ Condition ─INDICATES→ Treatment
   ├─PRESCRIBED→ Treatment ─MANAGES→ Condition
   ├─PRESENTS→ Symptom
   └─HAD_SURGERY→ Surgery
Guideline ─APPLIES_TO→ Condition,  ─RECOMMENDS→ Treatment
CareProtocol ─APPLIES_TO→ Condition, ─HAS_STEP→ ProtocolStep ─RECOMMENDS→ Treatment
```
