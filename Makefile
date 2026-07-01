# rmsai-relay — developer convenience targets.
#
# The vendored ecgtranscnn package (external/ecgtranscnn/) is gitignored and NOT a declared
# dependency, so `uv sync` does not track it and drops the editable install every time it runs.
# Always pair a sync with the editable re-install — that is what `make setup` does.

.DEFAULT_GOAL := setup
.PHONY: setup setup-all external test lint

# Base dev setup: sync core + dev deps, then (re)install the vendored package editable.
setup:
	uv sync --extra dev
	$(MAKE) external

# Full setup: all optional extras (rag, deid, voice, livekit, app) + the spaCy model Presidio needs.
setup-all:
	uv sync --extra dev --extra rag --extra deid --extra voice --extra livekit --extra app
	$(MAKE) external
	uv run python -m spacy download en_core_web_sm

# (Re)install the vendored ecgtranscnn editable. Run this after ANY `uv sync` you do by hand.
external:
	uv pip install -e external/ecgtranscnn

test:
	uv run pytest -q

lint:
	uv run ruff check .
