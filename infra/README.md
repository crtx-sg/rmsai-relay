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
before the first `docker-build`.

Optional backends stay behind profiles: `--profile llm` (containerized Ollama — skip it if
`ollama serve` runs on the host, both bind 11434), `--profile emr` (HAPI FHIR, binds 8080 like the
gateway), `--profile telemetry` (mosquitto). `--profile later` still selects all three.
