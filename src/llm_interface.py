from llm_sdk import Small_LLM_Model
from typing import Literal
from json import load
from .models import FunctionDefition, Prompt
from .trie import Trie
from .parsing import parsing
import math
import re


FUNCTION = 'F'
PARAMETERS = 'P'


class LlmInteface:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", *,
                 temperature: float = 0.3):
        infos = parsing()
        self.__functions: dict[str, FunctionDefition] = {f.name: f
                                                         for f in infos["functions"]}
        self.__prompts = self._format_prompt(infos.get("prompts"))
        self.__output: str = infos["output_file"]
        self.__temperature = temperature
        self.__model = Small_LLM_Model(model)
        if "Qwen3-0.6B" in model:
            end_key = self.__model.encode("<|im_end|>").squeeze().tolist()
        self.__end_key = end_key
        with open(self.__model.get_path_to_vocab_file()) as file:
            self.__vocab = {v: k for k, v in load(file).items()}
        print("="*60, f"\nVOCAB_SIZE:{len(self.__vocab)}\n{'='*60}")

    def _format_prompt(self, prompts: list[Prompt]) -> list[str]:
        formated_prompts = []
        for p in prompts:
            prompt = (f"<|im_start|>user\n{p!r}<|im_end|>"
                      "<|im_start|>assistant\n")
            formated_prompts.append(prompt)
        return formated_prompts

    def _sys_prompt(self, mode: Literal['F', 'P']) -> str:
        if mode == PARAMETERS:
            example = ('prompt: "What is the sum of 2 and 3?fn_add_numbers"\n'
                       'assistant: a=2, b=3\n'
                       'prompt: "Reverse the string \'hello\''
                       'fn_reverse_string\n'
                       'assistant: s=\'hello\'\n'
                       'prompt: "What is the square root of 16?'
                       'fn_get_square_root"\n'
                       'assistant: a=16\n')
            sys_prompt = ("<|im_start|>system\n"
                          "/no_think\n"
                          "Return only the paramaters of the funtion\n"
                          f"example: {example}\n"
                          "The functions you must use are the following: "
                          f"{list(self.__functions.values())!r}.\n"
                          "<|im_end|>")
        else:
            example = ('prompt: "What is the sum of 2 and 3?"\n'
                       'assistant: fn_add_numbers\n'
                       'prompt: "Reverse the string \'hello\"\n'
                       'assistant: fn_reverse_string\n'
                       'prompt: "What is the square root of 16?"\n'
                       'assistant: fn_get_square_root\n')
            sys_prompt = ("<|im_start|>system\n"
                          "/no_think\n"
                          "Return only the name of the funtion\n"
                          f"example: {example}\n"
                          "The functions you must use are the following: "
                          f"{list(self.__functions.values())!r}.\n"
                          "<|im_end|>")
        return sys_prompt

    def _softmax(self, logits: list[float]) -> list[float]:
        max_logit = max(logits)
        new = [math.exp((logit - max_logit) / self.__temperature)
               for logit in logits]
        logits_sum = sum(new)
        logits_sum = 1 if not logits_sum else logits_sum
        return [logit / logits_sum for logit in new]

    def _create_trie(self, tokens_list: list[list[int]]) -> Trie:
        root = Trie(-1)
        for tokens in tokens_list:
            tokens.append(self.__end_key)
            node = root
            for token in tokens:
                if token not in node.children:
                    node.add_children(Trie(token))
                node = node.children[token]
            node.is_end = True
        return root

    def _get_function_name(self, logits: list[float], trie: Trie) -> None:
        for index in range(len(logits)):
            if index in trie.children:
                continue
            elif index not in self.__vocab:
                logits[index] = float("-inf")
                continue
            else:
                logits[index] = float("-inf")

    def _generate_function(self, tokenized_prompt: list) -> str:
        trie = self._create_trie([self.__model.encode(
            func.name
            ).squeeze().tolist() for func in self.__functions.values()])
        function_name = list()
        while True:
            logits = self.__model.get_logits_from_input_ids(
                tokenized_prompt + function_name
                )
            self._get_function_name(logits, trie)
            logits = self._softmax(logits)
            token = logits.index(max(logits))
            if not trie.children or token == self.__end_key:
                break
            function_name.append(token)
            trie = trie.children[token]
        print(function_name)
        return self.__model.decode(function_name)

    def _get_parameters(self, logits: list[float],
                        allowed: list[str]) -> None:
        for index in range(len(logits)):
            if re.fullmatch(f"({'|'.join(re.escape(a) for a in allowed)})*",
                            self.__model.decode([index])):
                continue
            if index == self.__end_key:
                continue
            if index not in self.__vocab:
                logits[index] = float("-inf")
                continue
            else:
                logits[index] = float("-inf")

    def _get_parameter_state(self, parameters: list, func: FunctionDefition):
        if not parameters:
            return list(func.parameters.keys()) + ["-", ","]
        if any(parameters[-1] in self.__model.encode(param)
               for param in func.parameters.keys()):
            return ["="]
        if parameters[-1] == self.__model.encode("="):
            return ["\\w", "\\s" "\\", "\"", "\'", "?", "."]
        if parameters[-1] == self.__model.encode(','):
            return list(func.parameters.keys()) + ["-", ",", " "]
        else:
            return ["\\w", "\\s" "\\", "\"", "\'", "?", "."]

    def _generate_parameters(self, tokenized_prompt: list,
                             function_name: str) -> str:
        parameters = list()
        while True:
            logits = self.__model.get_logits_from_input_ids(
                tokenized_prompt + parameters
            )
            allowed = self._get_parameter_state(
                parameters,
                self.__functions[function_name]
            )
            print(allowed)
            self._get_parameters(logits, allowed)
            logits = self._softmax(logits)
            token = logits.index(max(logits))
            print(token, f"'{self.__model.decode(token)}'")
            if token == self.__end_key:
                break
            parameters.append(token)
            print(self.__model.decode(parameters))
        return self.__model.decode(parameters)

    def _generate_answer(self, tokenized_user_prompt: list) -> str:
        sys_prompt = self._sys_prompt(FUNCTION)
        tokenized_sys_prompt = self.__model.encode(
            sys_prompt
            ).squeeze().tolist()

        function_answer = self._generate_function(
            tokenized_sys_prompt + tokenized_user_prompt
            )
        print(function_answer)
        sys_prompt = self._sys_prompt(PARAMETERS)
        tokenized_sys_prompt = self.__model.encode(
                sys_prompt
                ).squeeze().tolist()
        tokenized_function_name = self.__model.encode(
            function_answer + "\n"
            ).squeeze().tolist()
        parameters_answer = self._generate_parameters(
                tokenized_sys_prompt +
                tokenized_user_prompt +
                tokenized_function_name,
                function_answer
                )
        print(parameters_answer)
        return

    def genrate(self) -> str:
        for p in self.__prompts:
            tokenized_user_prompt = self.__model.encode(p).squeeze().tolist()
            answer = self._generate_answer(tokenized_user_prompt)
            return
