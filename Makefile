# rmsai-relay — developer convenience targets.
#
# The vendored ecgtranscnn package (external/ecgtranscnn/) is gitignored and NOT a declared
# dependency, so `uv sync` does not track it and drops the editable install every time it runs.
# Always pair a sync with the editable re-install — that is what `make setup` does.
#
# Model weights are a separate step: ecgtranscnn gitignores `models/**/*.pt`, so a clone of it
# never carries checkpoints. `make weights` copies the real_v2 ensemble in from a local
# ecgtranscnn working copy (ECGTRANSCNN_DIR). Without it the pipeline runs the deterministic stub.

.DEFAULT_GOAL := setup
.PHONY: setup setup-all external weights test lint \
        docker-build docker-up docker-down docker-restart docker-logs docker-ps docker-shell \
        stores-up stores-down stores-check phone-up phone-down phone-logs demo-up demo-check demo-down

COMPOSE := docker compose -f infra/docker-compose.yml
# The backing services. These run in Docker even when the app runs on the host.
STORES := redis neo4j qdrant livekit
# Run app containers as the invoking user so files written into ./data are yours, not root's.
DOCKER_USER := APP_UID=$(shell id -u) APP_GID=$(shell id -g)

# Base dev setup: sync core + dev deps, then (re)install the vendored package editable.
setup:
	uv sync --extra dev
	$(MAKE) external

# Full setup: all optional extras (rag, deid, voice, livekit, app, pdf, llm-cloud) + the spaCy model
# Presidio needs.
setup-all:
	uv sync --extra dev --extra rag --extra deid --extra voice --extra livekit --extra app --extra pdf \
	    --extra llm-cloud
	$(MAKE) external
	uv run python -m spacy download en_core_web_sm

# (Re)install the vendored ecgtranscnn editable. Run this after ANY `uv sync` you do by hand.
external:
	uv pip install -e external/ecgtranscnn

# Copy the real-ECG model weights in from a local ecgtranscnn working copy (~42 MB: the 5-fold
# ensemble plus best_model.pt, the single-model fallback that .env.example documents).
# They are gitignored on both sides, so this is the reproducible form of "place the files manually".
# Override the source with: make weights ECGTRANSCNN_DIR=/path/to/ecgtranscnn
ECGTRANSCNN_DIR ?= ../ecgtranscnn
weights:
	@test -d "$(ECGTRANSCNN_DIR)/models/real_v2" || { \
	    echo "no real_v2 in $(ECGTRANSCNN_DIR)/models — set ECGTRANSCNN_DIR=/path/to/ecgtranscnn"; \
	    exit 1; }
	mkdir -p external/ecgtranscnn/models/real_v2
	cp $(ECGTRANSCNN_DIR)/models/real_v2/fold[0-4].pt \
	   $(ECGTRANSCNN_DIR)/models/real_v2/best_model.pt \
	   external/ecgtranscnn/models/real_v2/
	@ls -la external/ecgtranscnn/models/real_v2/*.pt

test:
	uv run pytest -q

lint:
	uv run ruff check .

# --- Hybrid: backing stores in Docker, app processes on the host ------------------------------
# Run the CLIs with `uv run python -m cli.<x>` yourself, but the stores still have to be up.
# `make docker-up` would ALSO start consumer/voice-worker/gateway, which collide with host-run
# copies on 8080/8081 — so this is the target to use when you drive the app by hand.

stores-up:
	$(COMPOSE) up -d $(STORES)

stores-down:
	$(COMPOSE) stop $(STORES)

# Which backing service is missing? Answers the question the driver tracebacks don't.
stores-check:
	@printf 'redis   : '; python3 -c "import socket;socket.create_connection(('localhost',6379),2).close();print('up')" 2>/dev/null || echo 'DOWN  -> make stores-up'
	@printf 'neo4j   : '; curl -fsS -m 2 -o /dev/null http://localhost:7474 && echo 'up' || echo 'DOWN  -> make stores-up'
	@printf 'qdrant  : '; curl -fsS -m 2 -o /dev/null http://localhost:6333/collections && echo 'up' || echo 'DOWN  -> make stores-up'
	@printf 'livekit : '; curl -fsS -m 2 -o /dev/null http://localhost:7880 && echo 'up' || echo 'DOWN  -> make stores-up'

# --- Docker: the whole relay (stores + consumer + voice worker + gateway) ----------------------
# Requires external/ecgtranscnn/ to be cloned (it is gitignored but baked into the image) and a
# .env at the repo root. See infra/docker-compose.yml for why the app services use host networking.

# Build the shared app image. Only needed once, and after pyproject.toml/uv.lock/vendored changes —
# the source is bind-mounted, so ordinary code edits need `docker-restart`, not a rebuild.
docker-build:
	$(COMPOSE) build

# Bring up everything: redis, neo4j, qdrant, livekit + consumer, voice-worker, gateway.
docker-up:
	$(DOCKER_USER) $(COMPOSE) up -d

# THE one command for the Docker demo: bring everything up in dependency order and prove it.
# `--wait` blocks until every service is running/healthy; compose starts the app services only once
# redis/neo4j/livekit report healthy, and restarts the voice worker whenever livekit is recreated.
# Then demo-check verifies endpoints and start order. Safe to re-run on a live stack.
demo-up:
	$(DOCKER_USER) $(COMPOSE) up -d --wait --wait-timeout 180
	@./infra/demo-check.sh

# Read-only: is the running demo healthy and started in the right order? Exit 0 = ready.
demo-check:
	@./infra/demo-check.sh

# Stop and remove every demo container (incl. the telephony phone worker). Volumes are kept, so the
# graph, vectors and KB survive; `make demo-up` brings it all back.
demo-down: docker-down

# --profile telephony: `down` skips services whose profile isn't active, which would leave the
# phone worker running.
docker-down:
	$(COMPOSE) --profile telephony down

# Phone-call worker on the telephony LiveKit (LIVEKIT_SIP_URL). See infra/docker-compose.yml.
phone-up:
	$(DOCKER_USER) $(COMPOSE) --profile telephony up -d voice-worker-phone

phone-down:
	$(COMPOSE) --profile telephony stop voice-worker-phone

phone-logs:
	$(COMPOSE) --profile telephony logs -f --tail=50 voice-worker-phone

# Pick up source edits (no rebuild needed — the repo is bind-mounted).
docker-restart:
	$(DOCKER_USER) $(COMPOSE) restart consumer voice-worker gateway

docker-logs:
	$(COMPOSE) logs -f --tail=50 consumer voice-worker gateway

docker-ps:
	$(COMPOSE) ps

# Interactive shell in the app image, with the repo mounted — for running any cli.* harness.
docker-shell:
	$(DOCKER_USER) $(COMPOSE) run --rm tools bash
