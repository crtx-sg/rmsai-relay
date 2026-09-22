# syntax=docker/dockerfile:1
#
# One shared image for every rmsai-relay application service (consumer, voice-worker, gateway,
# and the one-shot `tools` service). All of them run the same code with a different entrypoint, so
# a single image keeps one layer cache instead of three.
#
# The image carries only the *environment* — the Python venv with every extra installed, plus the
# vendored ecgtranscnn package and the spaCy model Presidio needs. The **source is bind-mounted**
# at /app by compose, so editing a file and restarting a container picks it up with no rebuild.
# Rebuild only when pyproject.toml / uv.lock / the vendored package change.
#
#   docker compose -f infra/docker-compose.yml build
#
# Build context is the repo root (see `build.context: ..`), because the vendored package lives at
# external/ecgtranscnn/. That directory is gitignored, so a fresh clone must run
# `git clone https://github.com/crtx-sg/ecgtranscnn external/ecgtranscnn` (pinned commit:
# bac4a01, see external/ecgtranscnn/PLACEHOLDER.md) before building. Model weights are NOT
# needed at build time — they are bind-mounted with the repo; see `make weights`.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# The venv lives OUTSIDE /app on purpose: compose bind-mounts the repo over /app at runtime, which
# would otherwise shadow an /app/.venv with the host's copy (built against the host interpreter).
# Everything therefore runs `python -m cli.<x>` off PATH, never `uv run` (which would re-sync into
# the shadowed /app/.venv).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Containers run as a non-root uid with no home directory; point every cache at a writable
    # path or torch/matplotlib/huggingface fail on first import with a permission error.
    HOME=/tmp \
    HF_HOME=/models/hf \
    MPLCONFIGDIR=/tmp/mpl

# libgomp1: onnxruntime (piper) + torch OpenMP runtime. curl: container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Dependencies. Cached until pyproject.toml / uv.lock change — the expensive layer (torch,
#    faster-whisper, sentence-transformers, presidio). `--no-install-project` because the project
#    itself is the bind mount, not a wheel.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra dev --extra rag --extra deid --extra voice --extra livekit --extra app --extra pdf

# 2. Vendored ecgtranscnn (model + simulator). Installed NON-editable, unlike the host `make
#    external`: the image must be importable on its own, and this is upstream code we don't edit.
#    The checkpoints and generated HDF5 under external/ecgtranscnn/{models,data} are NOT baked in —
#    they arrive with the bind mount and are referenced by path at runtime.
COPY external/ecgtranscnn /src/ecgtranscnn
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv /src/ecgtranscnn \
    && rm -rf /src

# 3. spaCy model for Presidio (DEID_BACKEND=presidio, or `auto` when presidio is installed — which
#    it is here). `spacy download` shells out to pip, which a uv-created venv does not ship.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv pip \
    && python -m spacy download en_core_web_sm

# 4. Model-cache mount point. Whisper (faster-whisper) and BGE (sentence-transformers) download
#    multi-hundred-MB weights on first use; compose backs this with a named volume so they survive
#    `down` and are shared by every service. Owned by the runtime uid, not root.
RUN mkdir -p /models/hf && chown -R 1000:1000 /models

# 5. Containers run as the *host's* uid (compose `user:`), which has no /etc/passwd entry here.
#    `getpass.getuser()` falls back to pwd.getpwuid() and raises `KeyError: getpwuid(): uid not
#    found`, which torch hits while resolving its inductor cache dir — i.e. `import torch` fails
#    outright. getuser() checks these env vars first, so setting them fixes it for ANY uid (a
#    baked-in useradd would only cover 1000). Kept last so it doesn't bust the dependency layers.
ENV USER=rmsai \
    LOGNAME=rmsai

# No source is copied: compose bind-mounts the repo at /app. Each service overrides `command`.
CMD ["python", "-c", "raise SystemExit('rmsai-relay: specify a command, e.g. python -m cli.consume')"]
