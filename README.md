*This project has been created as part of the 42 curriculum by jsouza.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling system that turns natural-language prompts
into structured, machine-executable function calls (name + typed arguments) using
a small local LLM (**Qwen3-0.6B**), without PyTorch, HuggingFace, or any prompt-and-hope
approach. The model never freely generates JSON: every token is validated at
generation time via **constrained decoding (logits masking)**, guaranteeing
100% valid, schema-compliant output even from a 600M-parameter model.

Given a set of function definitions and a list of prompts, the program produces
a JSON file mapping each prompt to the correct function name and arguments,
e.g. `"What is the sum of 2 and 3?"` → `fn_add_numbers(a=2.0, b=3.0)`.

## Instructions

### Requirements
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)
- `llm_sdk` package (provided, vendored under `llm_sdk/`)

### Install
```sh
make install
```

### Run
```sh
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```
Defaults (used when a flag is omitted):
- `--functions_definition data/input/functions_definition.json`
- `--input data/input/function_calling_tests.json`
- `--output data/output/function_calling_results.json`

### Makefile targets
| Target  | Description                                   |
|---------|------------------------------------------------|
| install | Creates `.venv` and syncs dependencies via `uv` |
| run     | Runs the program                                |
| lint    | Runs `flake8` and `mypy`                        |
| clean   | Removes `__pycache__` and `.mypy_cache`         |

## Algorithm Explanation

Constrained decoding is applied **per generation step**, not as a static filter:

1. The prompt is tokenized (ChatML format: `<|im_start|>system/user/assistant`,
   `<|im_end|>`, with `/no_think` to disable Qwen3's reasoning mode).
2. At each step, `get_logits_from_input_ids` returns logits for the full
   vocabulary.
3. A **whitelist mask** sets every invalid token's logit to `-inf`, based on
   the current generation state:
   - **Function name** — a `Trie` built from all valid function names (as
     token-id sequences) constrains the walk; only children of the current
     trie node stay unmasked.
   - **Parameters** — a state-aware regex/rule validator per type
     (`number`/`integer`/`float`, `string`, `boolean`) decodes each candidate
     token to text and checks it against the expected grammar (digits/dot for
     numbers, a boolean trie for `true`/`false`, character-class + regex-safety
     rules for strings, with extra constraints when the parameter is a regex
     itself).
4. Remaining logits are softmax-normalized (temperature-scaled, max-subtracted
   for numerical stability) and the highest-probability valid token is
   selected.
5. Generation stops on `<|im_end|>` or when no valid continuation remains.

The two-phase pipeline (function name first, then parameters conditioned on
the chosen function) is run once per prompt, and results are serialized to
JSON with `json.dump`.

## Design Decisions

- **Whitelist over blacklist**: enumerating the vocabulary and explicitly
  allowing valid tokens is safer than trying to blacklist every invalid one.
- **Trie-based constraint for function names**: since function names are a
  closed, known set, tree traversal is the natural constraint structure and
  guarantees only registered names can ever be produced.
- **Pydantic models** (`FunctionDefition`, `Prompt`, `Types`) validate input
  files against the expected schema before generation starts, with
  `extra='forbid'` to reject malformed definitions early.
- **BPE-aware validation**: because Qwen3 tokenizes in subwords, all
  validation functions operate on **decoded token strings**, not raw
  characters.

## Performance Analysis

- **JSON validity**: 100%, by construction — invalid tokens are never
  reachable during decoding.
- **Accuracy**: targets 90%+ correct function selection and argument
  extraction on the provided test set.
- **Speed**: designed to process the full test set within a few minutes on
  standard hardware; per-token vocabulary scans are the main cost.

## Challenges Faced

- **Double-escaping in string parameters**: the whitelist initially allowed
  bare backslashes for non-regex parameters, corrupting output — fixed by
  excluding `\` from the generic string whitelist.
- **Trie traversal bug**: children were checked against the root node
  instead of a stateful pointer — fixed by tracking `current_node`,
  advancing it per accepted token, and resetting on `is_end`.
- **Infinite generation loops**: a missing `continue` for the end-of-sequence
  token in the string-parameter validator prevented the model from ever
  stopping.
- **Tensor/scalar mismatch**: `.squeeze().tolist()` returned a bare `int` for
  single-token tensors — fixed by consistently using
  `encode(text).tolist()[0]`.
- **Vocab size mismatch**: the logits tensor could be larger than the vocab
  file — guarded with an explicit membership check before decoding an index.
- **Prompt leakage into parameters**: an insufficiently strict system prompt
  caused the model to reproduce memorized values instead of extracting them
  from the user prompt — fixed by tightening the parameter-extraction system
  prompt and few-shot examples.

## Testing Strategy

Manual testing against the provided `data/input/` samples, covering:
addition, greetings, string reversal, and regex-based substitution.
Edge cases exercised: empty strings, large numbers, and multi-parameter
functions. `flake8` and `mypy` (`--warn-return-any --warn-unused-ignores
--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`) are
run via `make lint` to enforce style and typing correctness.

## Example Usage

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Output (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Qwen3 documentation](https://qwenlm.github.io/)
- [ChatML format](https://github.com/openai/openai-python/blob/main/chatml.md)
- [Constrained decoding / guided generation overview](https://arxiv.org/abs/2307.09702)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

**AI usage**: AI assistance (Claude) was used to debug specific issues in the
constrained-decoding implementation (trie traversal state, logit-masking
regex rules, tensor/list shape mismatches) and to structure this README. All
generated suggestions were reviewed, tested, and understood before being
integrated; core algorithm design and implementation were done by the author.
