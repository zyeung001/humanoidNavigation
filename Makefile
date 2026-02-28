# Humanoid Navigation — Makefile
# Usage:
#   make setup                                   # Create venv and install deps
#   make train-standing                           # Train standing controller
#   make train-standing ARGS="--timesteps 5000000" # With custom args
#   make train-walking                            # Train walking controller
#   make train-walking ARGS="--debug --n-envs 1"  # Debug mode (no multiprocessing)
#   make record                                   # Record evaluation video
#   make clean                                    # Remove venv and caches

# ── Python detection ──────────────────────────────────────────────
# Prefer 3.11 > 3.12 > 3.10 > 3.13 > python3 (need >= 3.9 for gymnasium + SB3)
PYTHON := $(shell \
	for cmd in python3.11 python3.12 python3.10 python3.13 python3; do \
		if command -v $$cmd >/dev/null 2>&1; then echo $$cmd; break; fi; \
	done)
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python

ARGS ?=

# ── Targets ───────────────────────────────────────────────────────

.PHONY: setup train-standing train-walking record evaluate clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/installed ## Create virtualenv and install dependencies

$(VENV)/installed: requirements.txt
	@echo "Using Python: $(PYTHON)"
	@$(PYTHON) --version
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

train-standing: $(VENV)/installed ## Train standing controller (pass ARGS="..." for extra flags)
	$(PY) scripts/train_standing.py $(ARGS)

train-walking: $(VENV)/installed ## Train walking controller (pass ARGS="..." for extra flags)
	$(PY) scripts/train_walking.py $(ARGS)

record: $(VENV)/installed ## Record evaluation video (pass ARGS="..." for flags)
	$(PY) scripts/record_video.py $(ARGS)

evaluate: $(VENV)/installed ## Run evaluation script (pass ARGS="..." for flags)
	$(PY) scripts/evaluate.py $(ARGS)

clean: ## Remove virtualenv, caches, and generated files
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
