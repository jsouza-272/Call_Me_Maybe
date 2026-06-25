ifeq ($(shell [ -d ~/goinfre ] && echo true), true)
export UV_CACHE_DIR := ~/goinfre/jsouza/.cache/uv
export HF_HOME := ~/goinfre/.cache/huggingface
endif

run:
	clear
	uv build llm_sdk
	uv sync
	uv run python3 -m src