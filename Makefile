ifeq ($(shell [ -d $(HOME)/goinfre ] && echo true), true)
export UV_CACHE_DIR := $(HOME)/goinfre/jsouza/.cache/uv
export HF_HOME := $(HOME)/goinfre/.cache/huggingface
endif

UV_CHECK = $(shell .venv/bin/python3 -c 'import uv' 2>/dev/null && echo '1' || echo '0')
UV = .venv/bin/uv

install:
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
	fi
	@if [ "$(UV_CHECK)" = "0" ]; then \
		.venv/bin/pip install uv; \
	fi
	.venv/bin/uv sync;

run: install
	$(UV) run python3 -m src

clean:
	@rm -rf */__pycache__
	@rm -rf */*/__pycache__
	@rm -rf .mypy_cache

lint: install
	clear
	$(UV) run flake8
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

debug: install
	uv run python -m pdb -m src

.PHONY: lint install run debug