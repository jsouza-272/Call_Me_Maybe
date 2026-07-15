# Call Me Maybe

Function-calling pipeline built on top of `llm_sdk` and the `Qwen/Qwen3-0.6B` model.

## Requirements

- Python 3.10+
- `uv`

## Install

```bash
make install
```

## Run

```bash
make run
```

You can also override input/output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Quality checks

```bash
make lint
```