export UV_CACHE_DIR=/goinfre/jsouza/.cache/uv
export HF_HOME=/home/jsouza/goinfre/.cache/huggingface

run:
	uv build llm_sdk
	uv sync
	uv run python3 -m src