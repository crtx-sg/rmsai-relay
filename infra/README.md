# infra

Everything runs as a Docker service — the backing stores *and* the application processes.

| File | What it is |
|---|---|
| `docker-compose.yml` | all services; its header explains the profiles and why the app uses host networking |
| `Dockerfile` | one shared image for every app service (consumer, voice-worker, gateway, tools) |
| `livekit.yaml` | LiveKit dev config — binds all interfaces, advertises a browser-reachable RTC node IP |

```bash
make docker-build     # once, and after pyproject.toml / uv.lock / vendored-package changes
make docker-up        # redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway
make docker-logs      # tail the three app services
```

The source is **bind-mounted**, so a code edit needs `make docker-restart`, not a rebuild. The image
carries only the environment: the venv with every extra, the vendored `ecgtranscnn`, and the spaCy
model Presidio needs. `external/ecgtranscnn/` is gitignored but required at **build** time — clone it
before the first `docker-build` (pin: `bac4a01`).

Model **weights** are not baked into the image and are not needed to build it — `models/**/*.pt`
is gitignored upstream, so the clone carries none. Because the repo is bind-mounted, running
`make weights` on the host is enough for the containers to see them; set `ECG_CHECKPOINTS` in
`.env` to switch the services off the stub and onto the real model.

Optional backends stay behind profiles: `--profile llm` (containerized Ollama — skip it if
`ollama serve` runs on the host, both bind 11434), `--profile emr` (HAPI FHIR, binds 8080 like the
gateway), `--profile telemetry` (mosquitto). `--profile later` still selects all three.
