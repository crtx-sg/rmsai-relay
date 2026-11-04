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
.PHONY: setup setup-all external weights check-ecg test lint \
        docker-build docker-up docker-down docker-restart docker-logs docker-ps docker-shell \
        stores-up stores-down stores-check phone-up phone-down phone-logs phone-transcript demo-up demo-phone-up demo-check demo-down

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
# --python .venv: without it uv installs into an active conda env (CONDA_PREFIX) instead.
external:
	uv pip install --python .venv -e external/ecgtranscnn

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

# demo-up plus real phone calls: also starts the telephony phone worker (rmsai-agent-phone) and runs
# the consumer with `--transport sip`, so reportable events ring OUTBOUND_CALL_NUMBER instead of
# staging a WebRTC room. Compose doesn't read the repo .env for CONSUME_ARGS, hence it is set here.
# A later plain `make demo-up` recreates the consumer back on WebRTC.
# SMS fallback for an unanswered call: SignalWire texts from the same number that rings
# (DEMO_PHONE_NOTIFIER=simulated to send nothing).
DEMO_PHONE_NOTIFIER ?= signalwire
DEMO_PHONE_CONSUME_ARGS := --channel voice --caller livekit --transport sip --notifier $(DEMO_PHONE_NOTIFIER)
# Second hospital: with config/hospitals/h2.yaml present, demo-phone-up also brings up stack h2
# (compose project infra-h2, settings in infra/h2.env: own stores/consumer/workers/gateway on :8090,
# sharing h1's LiveKit), and the Simulate tab splits each run across both. HOSPITALS=h1 = h1 only.
, := ,
HOSPITALS ?= h1 $(if $(wildcard config/hospitals/h2.yaml),h2)
H2_ENV := set -a; . infra/h2.env; set +a;
H2_COMPOSE := $(COMPOSE) -p infra-h2
SIM_TARGETS := $(if $(filter h2,$(HOSPITALS)),h1=redis://localhost:6379/0$(,)h2=redis://localhost:6380/0)
SIM_CONSOLE := $(if $(filter h2,$(HOSPITALS)),h1)
APP_LINKS := $(if $(filter h2,$(HOSPITALS)),h1=http://localhost:8080$(,)h2=http://localhost:8090)

demo-phone-up:
	SIMULATE_TARGETS="$(SIM_TARGETS)" HOSPITAL_APPS="$(APP_LINKS)" SIMULATE_CONSOLE="$(SIM_CONSOLE)" CONSUME_ARGS="$(DEMO_PHONE_CONSUME_ARGS)" $(DOCKER_USER) \
		$(COMPOSE) --profile telephony up -d --wait --wait-timeout 180
	@./infra/demo-check.sh
	@if [ -n "$(filter h2,$(HOSPITALS))" ]; then $(MAKE) --no-print-directory h2-up H2_PHONE=1; fi

# Stack h2 only (h1's LiveKit must be up). Stores first, then the app services with --no-deps so
# compose doesn't start a second LiveKit in this project. H2_PHONE=1 adds the phone worker.
h2-up:
	$(H2_ENV) $(DOCKER_USER) $(H2_COMPOSE) up -d --wait --wait-timeout 180 redis neo4j qdrant
	$(H2_ENV) SIMULATE_TARGETS="$(SIM_TARGETS)" HOSPITAL_APPS="$(APP_LINKS)" SIMULATE_CONSOLE="$(SIM_CONSOLE)" CONSUME_ARGS="$(if $(H2_PHONE),$(DEMO_PHONE_CONSUME_ARGS))" \
		$(DOCKER_USER) $(H2_COMPOSE) $(if $(H2_PHONE),--profile telephony) up -d --wait \
		--wait-timeout 180 --no-deps consumer voice-worker gateway $(if $(H2_PHONE),voice-worker-phone)
	@$(H2_ENV) COMPOSE_PROJECT_NAME=infra-h2 ./infra/demo-check.sh

# One-time KB for a fresh h2 stack, BEFORE any h2 events (DEMO.md section 2; `index` never wipes).
h2-kb:
	$(H2_ENV) for c in "cli.graph migrate" "cli.graph protocols" "cli.graph extract --dir docs" \
		"cli.kb_vector --embedder bge index --dir docs"; do \
		$(DOCKER_USER) $(H2_COMPOSE) --profile tools run --rm --no-deps tools python -m $$c || exit 1; done

# Read-only: is the running demo healthy and started in the right order? Exit 0 = ready.
demo-check:
	@./infra/demo-check.sh

# Vendored ecgtranscnn at its pin, the same code on the host .venv and in the Docker image (and in
# running containers), and ECG_CHECKPOINTS present and identical to $(ECGTRANSCNN_DIR). Run after
# moving the pin, `make weights` or `make docker-build`.
check-ecg:
	@ECGTRANSCNN_DIR=$(ECGTRANSCNN_DIR) ./infra/check-ecg.sh

# Stop and remove every demo container (incl. the telephony phone worker). Volumes are kept, so the
# graph, vectors and KB survive; `make demo-up` brings it all back.
demo-down: docker-down

# --profile telephony: `down` skips services whose profile isn't active, which would leave the
# phone worker running.
docker-down:
	$(COMPOSE) --profile telephony down
	@if docker compose ls -aq | grep -qx infra-h2; then $(H2_ENV) $(H2_COMPOSE) --profile telephony down; fi

# Phone-call worker on the telephony LiveKit (LIVEKIT_SIP_URL). See infra/docker-compose.yml.
phone-up:
	$(DOCKER_USER) $(COMPOSE) --profile telephony up -d voice-worker-phone

phone-down:
	$(COMPOSE) --profile telephony stop voice-worker-phone

phone-logs:
	$(COMPOSE) --profile telephony logs -f --tail=50 voice-worker-phone

# The conversation only: STT transcripts, what the handler heard, the KB-grounded reply (logged
# before TTS speaks it), the greeting, PIN and spoken event alert. New lines only; Ctrl-C to stop.
# HOSPITAL=h1 (default) | h2 | all (both, each line prefixed [h1] / [h2]).
HOSPITAL ?= h1
TRANSCRIPT_RE := stt: ->|\[worker\] (heard|reply|speaking)|\[voice\]|graph query|declined
phone-transcript:
	@for h in $(if $(filter all,$(HOSPITAL)),h1 h2,$(HOSPITAL)); do \
		p=$$([ $$h = h1 ] && echo infra || echo infra-$$h); \
		docker compose -f infra/docker-compose.yml -p $$p --profile telephony ps --status running \
			-q voice-worker-phone | grep -q . || { echo "$$h: voice-worker-phone is not running \
			(make demo-phone-up)"; exit 1; }; done
	@for h in $(if $(filter all,$(HOSPITAL)),h1 h2,$(HOSPITAL)); do \
		p=$$([ $$h = h1 ] && echo infra || echo infra-$$h); \
		docker compose -f infra/docker-compose.yml -p $$p --profile telephony logs -f --tail=0 \
			--no-log-prefix voice-worker-phone 2>&1 | grep --line-buffered -E '$(TRANSCRIPT_RE)' \
			| sed -u "s/^/[$$h] /" & done; wait

# Pick up source edits (no rebuild needed — the repo is bind-mounted): the app services and phone
# worker of h1 and, if it exists, h2. `up -d` (demo-phone-up) does not: it only recreates a container
# whose compose settings changed. Refuses while a call is up (a restart mid-call leaves the phone
# connected to a dead worker); FORCE=1 overrides.
STACKS = infra $(shell docker compose ls -aq 2>/dev/null | grep -x infra-h2)
docker-restart:
	@for p in $(STACKS); do \
		a=$$(docker compose -f infra/docker-compose.yml -p $$p exec -T redis redis-cli get rmsai:calls:active 2>/dev/null); \
		[ -z "$$a" ] || { echo "$$p: call in progress (event $$a): restart after it ends, or FORCE=1"; \
		[ -n "$(FORCE)" ] || exit 1; }; done
	@for p in $(STACKS); do echo "restarting $$p"; \
		docker compose -f infra/docker-compose.yml -p $$p --profile telephony restart \
		consumer voice-worker voice-worker-phone gateway || exit 1; done
	@./infra/demo-check.sh
	@if echo "$(STACKS)" | grep -qw infra-h2; then $(H2_ENV) COMPOSE_PROJECT_NAME=infra-h2 ./infra/demo-check.sh; fi

docker-logs:
	$(COMPOSE) logs -f --tail=50 consumer voice-worker gateway

docker-ps:
	$(COMPOSE) ps

# Interactive shell in the app image, with the repo mounted — for running any cli.* harness.
docker-shell:
	$(DOCKER_USER) $(COMPOSE) run --rm tools bash
