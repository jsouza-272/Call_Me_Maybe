"""LLM integration and constrained decoding helpers."""

from __future__ import annotations

import json
import os
from typing import Any, cast

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, Prompt, Types
from .parsing import parsing
from .trie import Trie


END_OF_MESSAGE_TOKEN = "<|im_end|>"


class LlmInterface:
    """Generate structured function calls from natural language prompts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        max_tokens: int = 128,
    ) -> None:
        parsed_data = parsing()
        self._functions: dict[str, FunctionDefinition] = {
            function.name: function
            for function in parsed_data["functions"]
        }
        self._prompts: list[Prompt] = parsed_data["prompts"]
        self._output_file: str = parsed_data["output_file"]
        self._max_tokens = max_tokens
        self._model = Small_LLM_Model(model_name)
        self._end_token_id = self._get_end_token_id()

    def _get_end_token_id(self) -> int:
        tokenized = cast(
            list[int],
            self._model.encode(END_OF_MESSAGE_TOKEN).tolist()[0],
        )
        if not tokenized:
            raise ValueError("Unable to resolve end-of-message token id")
        return tokenized[0]

    def _chat_prompt(self, system_message: str, user_message: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{system_message}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _encode(self, text: str) -> list[int]:
        return cast(list[int], self._model.encode(text).tolist()[0])

    def _create_trie(
        self, tokenized_sequences: list[list[int]], *, append_end: bool = False
    ) -> Trie:
        root = Trie(-1)
        for sequence in tokenized_sequences:
            tokens = sequence.copy()
            if append_end:
                tokens.append(self._end_token_id)
            node = root
            for token in tokens:
                if token not in node.children:
                    node.add_child(Trie(token))
                node = node.children[token]
            node.is_end = True
        return root

    def _pick_best_token(self, logits: list[float], allowed: set[int]) -> int:
        best_token = -1
        best_logit = float("-inf")
        for token in allowed:
            score = logits[token]
            if score > best_logit:
                best_logit = score
                best_token = token
        if best_token < 0:
            raise RuntimeError(
                "No valid token available during constrained decode"
            )
        return best_token

    def _generate_with_trie(self, prompt_tokens: list[int], trie: Trie) -> str:
        generated: list[int] = []
        node = trie
        for _ in range(self._max_tokens):
            logits = self._model.get_logits_from_input_ids(
                prompt_tokens + generated
            )
            allowed = set(node.children.keys())
            if not allowed:
                break
            token = self._pick_best_token(logits, allowed)
            if token == self._end_token_id:
                break
            generated.append(token)
            node = node.children[token]
        return self._model.decode(generated).strip()

    def _generate_text(
        self, prompt_tokens: list[int], *, stop_on_json: bool
    ) -> str:
        generated: list[int] = []
        for _ in range(self._max_tokens):
            logits = self._model.get_logits_from_input_ids(
                prompt_tokens + generated
            )
            token = max(range(len(logits)), key=logits.__getitem__)
            if token == self._end_token_id:
                break
            generated.append(token)
            if stop_on_json:
                current_text = self._model.decode(generated)
                if self._looks_like_completed_json(current_text):
                    break
        return self._model.decode(generated).strip()

    def _looks_like_completed_json(self, text: str) -> bool:
        if "{" not in text or "}" not in text:
            return False
        return (
            text.count("{") == text.count("}")
            and text.rstrip().endswith("}")
        )

    def _select_function_name(self, user_prompt: str) -> str:
        functions_summary = "\n".join(
            f"- {function.name}: {function.description}"
            for function in self._functions.values()
        )
        system_message = (
            "/no_think\n"
            "Select exactly one function name from the list below.\n"
            "Return only the function name and nothing else.\n"
            f"{functions_summary}"
        )
        prompt = self._chat_prompt(system_message, user_prompt)
        prompt_tokens = self._encode(prompt)
        encoded_names: list[list[int]] = []
        for name in self._functions.keys():
            encoded_names.append(self._encode(name))
            encoded_names.append(self._encode(f" {name}"))
        trie = self._create_trie(encoded_names, append_end=True)
        selected_name = self._generate_with_trie(prompt_tokens, trie).strip()
        if selected_name in self._functions:
            return selected_name
        normalized = selected_name.strip()
        if normalized in self._functions:
            return normalized
        for known_name in self._functions.keys():
            if known_name in normalized:
                return known_name
        return next(iter(self._functions))

    def _extract_parameters_json(
        self, user_prompt: str, function_definition: FunctionDefinition
    ) -> dict[str, Any]:
        schema_json = json.dumps(
            {
                name: details.type
                for name, details in function_definition.parameters.items()
            },
            ensure_ascii=False,
        )
        system_message = (
            "/no_think\n"
            "Extract parameters for the selected function.\n"
            "Return ONLY a JSON object with exactly these keys "
            "and inferred values.\n"
            "No markdown, no explanations.\n"
            f"Parameter schema: {schema_json}"
        )
        user_message = (
            f"Selected function: {function_definition.name}\n"
            f"User prompt: {user_prompt}"
        )
        prompt = self._chat_prompt(system_message, user_message)
        raw_answer = self._generate_text(
            self._encode(prompt),
            stop_on_json=True,
        )
        return self._parse_json_object(raw_answer)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        candidate = text.strip()
        if not candidate:
            return {}
        parsed = self._try_parse_json(candidate)
        if parsed is not None:
            return parsed
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        parsed = self._try_parse_json(candidate[start:end + 1])
        return parsed if parsed is not None else {}

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _default_for_type(self, expected_type: str) -> Any:
        if expected_type in {"number", "float"}:
            return 0.0
        if expected_type == "integer":
            return 0
        if expected_type == "boolean":
            return False
        return ""

    def _coerce_parameter(self, value: Any, expected: Types) -> Any:
        if value is None:
            return self._default_for_type(expected.type)
        try:
            if expected.type in {"number", "float"}:
                return float(value)
            if expected.type == "integer":
                return int(float(value))
            if expected.type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                lowered = str(value).strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
                return False
            return str(value)
        except (TypeError, ValueError):
            return self._default_for_type(expected.type)

    def _generate_answer_for_prompt(self, user_prompt: str) -> dict[str, Any]:
        function_name = self._select_function_name(user_prompt)
        function_definition = self._functions[function_name]
        raw_parameters = self._extract_parameters_json(
            user_prompt,
            function_definition,
        )

        parameters: dict[str, Any] = {}
        for parameter_name, parameter_type in (
            function_definition.parameters.items()
        ):
            parameters[parameter_name] = self._coerce_parameter(
                raw_parameters.get(parameter_name), parameter_type
            )
        return {
            "prompt": user_prompt,
            "name": function_name,
            "parameters": parameters,
        }

    def generate(self) -> list[dict[str, Any]]:
        """Generate function-calling outputs for all loaded prompts."""

        return [
            self._generate_answer_for_prompt(prompt.prompt)
            for prompt in self._prompts
        ]

    def save_json(self, results: list[dict[str, Any]]) -> None:
        """Persist generated outputs in the configured output path."""

        output_dir = os.path.dirname(self._output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self._output_file, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=4)

    # Backward-compatible method names used by older code.
    def genrate(self) -> list[dict[str, Any]]:
        return self.generate()


# Backward-compatible alias for existing imports.
LlmInteface = LlmInterface
