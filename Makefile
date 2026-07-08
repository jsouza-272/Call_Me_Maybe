ifeq ($(shell [ -d $(HOME)/goinfre ] && echo true), true)
export UV_CACHE_DIR := $(HOME)/goinfre/jsouza/.cache/uv
export HF_HOME := $(HOME)/goinfre/.cache/huggingface
endif

run:
	clear
	uv sync
	uv run python3 -m src --input data/input/string_prompts.json
test:
	clear
	uv sync
	uv run python3 src/.test.py