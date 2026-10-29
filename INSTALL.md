# Install

How to install rmsai-relay and bring it up. Background: [README.md](README.md); settings: [CONFIGURATION.md](CONFIGURATION.md); day-to-day running: [OPERATIONS.md](OPERATIONS.md).

## Setup

Two ways to run it. **Docker** brings the whole relay up in one command and needs no host Python;
**host** installs the environment locally, which is the better loop for editing and running tests.
They share the same `.env` and the same datastores, so you can mix them.

### A. Everything in Docker (recommended for a demo)

```bash
# 1. Vendor the ECG model + simulator (gitignored; baked into the image, so clone it FIRST)
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn
git -C external/ecgtranscnn checkout a843ccf

# 1b. Model weights (optional). They are gitignored upstream, so the clone carries none — copy
#     them from a local ecgtranscnn working copy that has them. Without this the pipeline still
#     runs, on the deterministic stub.
make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn   # then set ECG_CHECKPOINTS in .env

# 2. Configure
cp .env.example .env      # then set LIVEKIT_API_KEY / LIVEKIT_API_SECRET, HOSPITAL_ID, …

# 3. Build the shared app image (once; ~5-10 min for torch + whisper + presidio)
make docker-build

# 4. Bring up every service, in dependency order, then check it (READY = all ok)
make demo-up
```

![Docker services, cloud services and call flow](deployment.png)

*What runs where (regenerate with `uv run python docs/make_deploy_diagram.py`); the logical
pipeline view is [`architecture.png`](architecture.png).*

That starts **redis, neo4j, qdrant, livekit** and the three application services — **consumer**
(bus → persist → dispatch), **voice-worker** (the LiveKit agent), and **gateway** (the companion
app on `http://localhost:8080/`). `make docker-ps` shows the state, `make docker-logs` tails the
three app services.

The **source is bind-mounted**, so a code edit needs only `make docker-restart` — no rebuild.
Rebuild (`make docker-build`) only when `pyproject.toml`, `uv.lock`, or the vendored package
changes.

Run any CLI harness in the same image, with no host Python:

```bash
docker compose -f infra/docker-compose.yml run --rm tools \
    python -m cli.ingest --file data/inference/<f>.h5 --emit bus
    # uses ECG_CHECKPOINTS from .env; override per-run with
    #   --checkpoint external/ecgtranscnn/models/real_v2/fold{0,1,2,3,4}.pt
make docker-shell        # or an interactive shell in that image
```

> **App services use `network_mode: host`.** That is deliberate: it makes every `localhost` URL in
> `.env` resolve the same inside a container as on the host, keeps an Ollama running on the *host*
> reachable, and — the reason that matters most — lets LiveKit's advertised `rtc.node_ip: 127.0.0.1`
> mean the same thing to the voice worker as to the browser. On a bridge network the worker would
> receive that candidate, try to connect to itself, and audio would never flow even though
> signaling succeeded. The tradeoff is no port isolation, and Linux-only portability. See the header
> of `infra/docker-compose.yml`.

Two services stay behind profiles because they collide with this setup: `--profile llm`
(`model-server`, containerized Ollama — **skip it if `ollama serve` already runs on the host**;
both bind 11434) and `--profile emr` (`hapi-fhir` — binds 8080, same as the gateway; move the
gateway with `GATEWAY_PORT`). `--profile telemetry` adds mosquitto. The legacy `--profile later`
still selects those three. Two more profiles hold on-demand services: `--profile tools` (`tools`,
the one-shot CLI runner, and `real-samples`, see [real ECG](OPERATIONS.md#curating-a-held-out-real-ecg-demo-set-clireal_samples);
`docker compose run` enables it implicitly) and `--profile telephony` (`voice-worker-phone`, started
by `make phone-up`, see [Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire)).

#### Two `.env` readers

The **app** (every `python -m cli.*`, on the host or in a container) reads the repo-root `.env`
itself; exported shell variables win over it. **Docker Compose** does *not* read that file for its
`${…}` settings: with `-f infra/docker-compose.yml` its project directory is `infra/`, which has no
`.env`. So Compose-level knobs in the root `.env` are silently ignored. Export them in the shell or
pass them inline:

```bash
CONSUME_ARGS="--channel voice --caller livekit --transport webrtc" make docker-up   # consumer's CLI args (this is the default)
GATEWAY_PORT=8090 make docker-up
ECGPKG_HOST_DIR=/abs/path/to/packages docker compose -f infra/docker-compose.yml run --rm real-samples list
```

Compose-level keys: `CONSUME_ARGS`, `GATEWAY_PORT`, `CONSUMER_HF_OFFLINE` / `WORKER_HF_OFFLINE` (set `0` on a fresh `hf_cache`), `ECGPKG_HOST_DIR`, `ECGPKG_NAME`,
`APP_UID`/`APP_GID`, `LIVEKIT_WORKER_HTTP_PORT` (container), `LIVEKIT_SIP_WORKER_HTTP_PORT`
(container), `NEO4J_USER`/`NEO4J_PASSWORD` (the neo4j container's auth) and the `*_PORT` host-port
overrides. Changing `NEO4J_PASSWORD` in `.env` alone therefore changes what the app sends but not
what the container expects.

### B. On the host

```bash
# 1. Vendor the ECG model + simulator (gitignored; see external/ecgtranscnn/PLACEHOLDER.md)
git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn
git -C external/ecgtranscnn checkout a843ccf
#    Weights are gitignored upstream; copy them from a local ecgtranscnn working copy:
#    make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn   (skip it to run on the stub)

# 2. Install everything (recommended for a demo). `make setup-all` = all extras
#    (rag, deid, voice, livekit, app, pdf, llm-cloud) + the vendored ecgtranscnn editable + the spaCy model.
make setup-all
#    Lean alternative (core + dev only): `make setup`.
#    Re-run `make external` after ANY hand-run `uv sync` — the vendored package is gitignored, so
#    `uv sync` uninstalls it and it must be reinstalled editable.

# 3. Bring up the backing stores only (leave the app services to your terminals)
make stores-up            # = docker compose … up -d redis neo4j qdrant livekit
make stores-check         # which ones are actually reachable?

# 4. Run tests (offline; the full suite needs the stores; graph-wiping tests skip unless RMSAI_ALLOW_LIVE_GRAPH_RESET=1)
uv run pytest -q -m "not infra"
```

> **The stores run in Docker even in this mode.** Running the CLIs with `uv run` does *not* make
> them self-contained — `cli.ingest --emit bus` still needs Redis, `cli.consume` needs Redis + Neo4j
> + Qdrant, and the app needs LiveKit. A `docker compose down` removes them all, and the next CLI
> run fails with `Error 111 connecting to localhost:6379`. `make stores-up` is the fix;
> `make stores-check` tells you which one is missing:
>
> ```
> redis   : up
> neo4j   : DOWN  -> make stores-up
> qdrant  : up
> livekit : up
> ```
>
> Use `make stores-up`, **not** `make docker-up`, when you drive the app by hand — the latter also
> starts consumer/voice-worker/gateway, which collide with host-run copies on 8080/8081.

### Make targets

| Target | What it does |
|---|---|
| `make setup` | `uv sync --extra dev` + `make external` (core + dev only) |
| `make setup-all` | all extras (rag, deid, voice, livekit, app, pdf, llm-cloud) + `make external` + spaCy `en_core_web_sm` |
| `make external` | (re)install the vendored `external/ecgtranscnn` editable — run after any manual `uv sync` |
| `make weights` | copy the `real_v2` checkpoints in from a local ecgtranscnn working copy (`ECGTRANSCNN_DIR`, default `../ecgtranscnn`) — they are gitignored, so no clone carries them |
| `make stores-up` / `make stores-down` | start / stop **only** the backing stores (redis, neo4j, qdrant, livekit) — the target to use when you run the CLIs on the host |
| `make stores-check` | which backing stores are reachable, and what to run if one is down |
| `make docker-build` | build the shared app image (`infra/Dockerfile`) |
| `make docker-up` / `make docker-down` | start every default service / stop everything (including the `telephony` phone worker). `make demo-down` is the same as `docker-down` |
| `make docker-restart` | restart the three app services to pick up source edits (no rebuild) |
| `make docker-logs` / `make docker-ps` | tail the app services / show container state |
| `make docker-shell` | interactive shell in the app image with the repo mounted |
| `make demo-up` / `make demo-check` | bring up every default service in dependency order and health-check it (the consumer stages WebRTC rooms: **no phone rings**) / re-run only the check |
| `make demo-phone-up` | `demo-up` plus real phone calls: also starts the phone worker and runs the consumer with `--transport sip --notifier signalwire`, so calls go through the Redis call queue and an unanswered call (including a pickup with no PIN) is texted (`DEMO_PHONE_NOTIFIER=simulated` to send nothing). A later plain `make demo-up` puts the consumer back on WebRTC |
| `make phone-up` / `make phone-down` / `make phone-logs` | start / stop / tail the phone-call worker (`voice-worker-phone`, profile `telephony`) |
| `make phone-transcript` | follow the phone worker's conversation only: STT transcript, what was heard, the graph query, the reply (before TTS), the PIN and the spoken alert |
| `make test` / `make lint` | `uv run pytest -q` / `ruff check .` |

### Optional extras (à la carte, if you skipped `make setup-all`)

```bash
uv sync --extra rag                                   # real BGE embeddings + reranker
uv sync --extra deid && uv run python -m spacy download en_core_web_sm   # Presidio de-id
uv sync --extra voice                                 # faster-whisper STT + Piper TTS
uv sync --extra livekit                               # LiveKit agent worker + SIP/WebRTC
uv sync --extra app                                   # companion-app gateway (FastAPI/uvicorn)
uv sync --extra pdf                                   # PDF uploads for cli.kb_upload (pypdf)
uv sync --extra llm-cloud                             # anthropic / openai SDKs for cloud LLM providers
# NOTE: any bare `uv sync` uninstalls the vendored ecgtranscnn — follow with `make external`.
```

`.env` keys that matter: `REDIS_URL`, `NEO4J_*`, `QDRANT_URL`, `DEID_BACKEND`, `STT_BACKEND`/
`TTS_BACKEND`, `INBOUND_AUTH_PIN`, `AUDIO_WAKE_REQUIRED` (wake-word gate on/off),
`INBOX_SPEAK_ON_SELECT` (voice the event on worklist select), `LIVEKIT_URL` / `LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET` + `HOSPITAL_ID` for the app and WebRTC calls (`HOSPITAL_ID` also selects
`config/hospitals/<id>.yaml`: dispatch mode, phone numbers, alert thresholds), and the
telephony block for phone calls (see [Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire)).

**Swappable speech backends.** STT/TTS sit behind `STTAdapter`/`TTSAdapter` (`voice/adapters.py`),
selected by `STT_BACKEND` / `TTS_BACKEND`: self-hosted **whisper**/**piper** (set them explicitly;
the code default is `stub`, which produces no real audio and makes the worker warn), or cloud
**elevenlabs** (STT "Scribe" + TTS; stdlib HTTP, no extra dep) for **accuracy/latency benchmarking**.
Set `ELEVENLABS_API_KEY` (+ optional `ELEVENLABS_VOICE_ID` / `ELEVENLABS_TTS_MODEL` /
`ELEVENLABS_STT_MODEL`). Compare backends + latency offline with
`uv run python -m cli.speech_check --tts elevenlabs --stt elevenlabs` (TTS → STT round-trip, prints
per-leg ms).

PHI handling differs by direction (the two cloud legs are **not** symmetric):

- **Cloud TTS is PHI-guarded.** Every spoken string (greeting, alert, answers) is run through the
  configured de-identifier (`DEID_BACKEND`: Presidio/regex) by `DeidentifyingTTS` **before** the
  text leaves the host — on top of pseudonym-by-construction. So no name/SSN/etc. reaches the
  provider; cloud TTS is safe for the de-identified clinical text this system produces.
- **Cloud STT cannot be pre-redacted.** It sends **raw caller audio** to be transcribed, so there is
  nothing to de-identify first (de-id needs text). If a clinician *speaks* an identifier it reaches
  the provider. Therefore `STT_BACKEND=elevenlabs` is **synthetic-speech only, never real PHI**
  (hard rules #4/#5). The worker prints a warning. The self-hosted whisper path keeps STT on-box.

> If you hit `ModuleNotFoundError: presidio_analyzer`, set `DEID_BACKEND=auto` (or `regex`), or
> install the `deid` extra above.
