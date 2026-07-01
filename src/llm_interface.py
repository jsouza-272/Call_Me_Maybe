from llm_sdk import Small_LLM_Model
from typing import Literal
from json import load
from .models import FunctionDefition, Prompt
from .trie import Trie
from .parsing import parsing
from .states import ParameterState
import math
import re
import sys


FUNCTION = 'F'
PARAMETERS = 'P'


class LlmInteface:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", *,
                 temperature: float = 0.3):
        infos = parsing()
        self.__functions: dict[str, FunctionDefition] = {
            f.name: f
            for f in infos["functions"]
            }
        self.__prompts = infos.get("prompts")
        self.__output: str = infos["output_file"]
        self.__temperature = temperature
        self.__model = Small_LLM_Model(model)
        if "Qwen3-0.6B" in model:
            end_key = self.__model.encode("<|im_end|>").squeeze().tolist()
        self.__end_key = end_key
        with open(self.__model.get_path_to_vocab_file()) as file:
            self.__vocab = {v: k for k, v in load(file).items()}
        print("="*60, f"\nVOCAB_SIZE:{len(self.__vocab)}\n{'='*60}")

    def _format_prompt(self, prompt: str,
                       function_answer: str = "") -> list[str]:
        if not function_answer:
            prompt = (f"<|im_start|>user\n{prompt!r}<|im_end|>"
                      "<|im_start|>assistant\n")
        else:
            prompt = (f"<|im_start|>user\n{prompt!r}"
                      f"{function_answer}<|im_end|>"
                      "<|im_start|>assistant\n")
        return prompt

    def _sys_prompt(self, mode: Literal['F', 'P']) -> str:
        if mode == PARAMETERS:
            example = ('prompt: "What is the sum of 2 and 3?fn_add_numbers"\n'
                       'assistant: a=2.0, b=3.0\n'
                       'prompt: "Reverse the string \'hello\''
                       'fn_reverse_string\n'
                       'assistant: s=\'hello\'\n'
                       'prompt: "What is the square root of 16?'
                       'fn_get_square_root"\n'
                       'assistant: a=16.0\n')
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
        return self.__model.decode(function_name)

    def _get_parameters(self, logits: list[float],
                        state: ParameterState, trie: Trie) -> None:
        for index in range(len(logits)):
            match state:
                case ParameterState.PARAMETER:
                    if index in trie.children:
                        continue
                    logits[index] = float("-inf")
                case ParameterState.EQUAL:
                    if (index == self.__model.encode("=").squeeze().tolist()):
                        continue
                    logits[index] = float("-inf")
                case ParameterState.VALUENUMBER:
                    if (re.fullmatch(
                        r"[\d .,]*", self.__model.decode([index])
                        ) and index in self.__vocab
                            or index == self.__end_key):
                        continue
                    logits[index] = float("-inf")
                case ParameterState.VALUESTRING:
                    if (re.fullmatch(
                        r"[A-Za-z.,?!\\/\"\']", self.__model.decode([index])
                        ) and index in self.__vocab
                            or index == self.__end_key):
                        continue
                    logits[index] = float("-inf")

    def _get_parameter_state(self, parameters: list,
                             func: FunctionDefition) -> ParameterState:
        if not parameters:
            return ParameterState.PARAMETER
        decode_param = self.__model.decode(parameters)
        if decode_param.endswith(', '):
            return ParameterState.PARAMETER
        if (decode_param.split()[-1] in func.parameters
                and not decode_param.endswith('=')):
            return ParameterState.EQUAL
        if all(param_type.type == "number"
               for param_type in func.parameters.values()):
            return ParameterState.VALUENUMBER
        return ParameterState.VALUESTRING

    def _generate_parameters(self, tokenized_prompt: list,
                             function_name: str) -> str:
        list_parameter = []
        for param in self.__functions[function_name].parameters:
            tokenized_param = self.__model.encode(param).squeeze().tolist()
            if not isinstance(tokenized_param, list):
                tokenized_param = [tokenized_param]
            list_parameter.append(tokenized_param)
        trie = self._create_trie(list_parameter)
        parameters = list()
        while True:
            logits = self.__model.get_logits_from_input_ids(
                tokenized_prompt + parameters
            )
            state = self._get_parameter_state(
                parameters,
                self.__functions[function_name]
            )
            print(state)
            self._get_parameters(logits, state, trie)
            logits = self._softmax(logits)
            token = logits.index(max(logits))
            if token == self.__end_key:
                print(f'end: {self.__model.decode(parameters)}')
                break
            if token in trie.children and not trie.is_end:
                trie = trie.children[token]
            parameters.append(token)
            print(self.__model.decode(parameters))
        return self.__model.decode(parameters)

    def _generate_answer(self, user_prompt: str) -> str:
        sys_prompt = self._sys_prompt(FUNCTION)
        formated_user_prompt = self._format_prompt(user_prompt)
        tokenized_user_prompt = self.__model.encode(
            formated_user_prompt
        ).squeeze().tolist()
        tokenized_sys_prompt = self.__model.encode(
            sys_prompt
            ).squeeze().tolist()

        function_answer = self._generate_function(
            tokenized_sys_prompt + tokenized_user_prompt
            )
        print(function_answer)

        formated_user_prompt = self._format_prompt(
            user_prompt, function_answer
        )
        tokenized_user_prompt = self.__model.encode(
            formated_user_prompt
        ).squeeze().tolist()
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
            answer = self._generate_answer(p)
            return
