ifeq ($(shell [ -d $(HOME)/goinfre ] && echo true), true)
export UV_CACHE_DIR := $(HOME)/goinfre/jsouza/.cache/uv
export HF_HOME := $(HOME)/goinfre/.cache/huggingface
endif

UV = .venv/bin/uv

install:
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
	fi
	@if [ ! -x ".venv/bin/uv" ]; then \
		.venv/bin/pip install uv; \
	fi
	@$(UV) sync

run: install
	$(UV) run python -m src

debug: install
	$(UV) run python -m pdb -m src

clean:
	@rm -rf */__pycache__
	@rm -rf */*/__pycache__
	@rm -rf */*/*/__pycache__
	@rm -rf .mypy_cache

lint: install
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict: install
	$(UV) run flake8 .
	$(UV) run mypy . --strict

.PHONY: clean debug install lint lint-strict run