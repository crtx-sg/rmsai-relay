# infra

Everything runs as a Docker service — the backing stores *and* the application processes.

| File | What it is |
|---|---|
| `docker-compose.yml` | all services; its header explains the profiles and why the app uses host networking |
| `Dockerfile` | one shared image for every app service (consumer, voice-worker, gateway, tools, real-samples, voice-worker-phone) |
| `livekit.yaml` | LiveKit dev config — binds all interfaces, advertises a browser-reachable RTC node IP |

```bash
make docker-build     # once, and after pyproject.toml / uv.lock / vendored-package changes
make docker-up        # redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway
make docker-logs      # tail the three app services
```

The source is **bind-mounted**, so a code edit needs `make docker-restart` (consumer, voice-worker,
gateway; for the phone worker: `make phone-down phone-up`), not a rebuild. The image
carries only the environment: the venv with every extra, the vendored `ecgtranscnn`, and the spaCy
model Presidio needs. `external/ecgtranscnn/` is gitignored but required at **build** time — clone it
before the first `docker-build` (pin: `bac4a01`).

Model **weights** are not baked into the image and are not needed to build it — `models/**/*.pt`
is gitignored upstream, so the clone carries none. Because the repo is bind-mounted, running
`make weights` on the host is enough for the containers to see them; set `ECG_CHECKPOINTS` in
`.env` to switch the services off the stub and onto the real model.

Optional backends stay behind profiles: `--profile llm` (containerized Ollama — skip it if
`ollama serve` runs on the host, both bind 11434), `--profile emr` (HAPI FHIR, binds 8080 like the
gateway), `--profile telemetry` (mosquitto). `--profile later` still selects all three. Two more
profiles hold on-demand services:
- `--profile tools`: `tools` (one-shot CLI runner) and `real-samples`, which bind-mounts
  `${ECGPKG_HOST_DIR:-../../ecg_sigma/packages}` read-only at `/ecgpkg` and fails if it's missing
  (runbook: `DEMO.md`);
- `--profile telephony`: `voice-worker-phone`, via `make phone-up` / `phone-down` / `phone-logs`
  (`make docker-down` stops it too).

**`.env` caveat.** With `-f infra/docker-compose.yml` Compose's project directory is `infra/`, so it
does **not** read the repo-root `.env` for `${…}` settings. The app containers read `/app/.env`
themselves, but Compose-level knobs must be exported in the shell or passed inline
(`CONSUME_ARGS="…" make docker-up`): `CONSUME_ARGS`, `ECGPKG_HOST_DIR`, `ECGPKG_NAME`,
`GATEWAY_PORT`, `APP_UID/GID`, `LIVEKIT_WORKER_HTTP_PORT`, `LIVEKIT_SIP_WORKER_HTTP_PORT`,
`NEO4J_USER/PASSWORD` and the `*_PORT` overrides. A value in a service's `environment:` overrides the
same key in `.env`; changing `NEO4J_PASSWORD` only in `.env` breaks auth against the container.

Other targets: `make stores-up` / `stores-down` / `stores-check` (backing stores only, for host
runs), `docker-ps`, `docker-shell`.
