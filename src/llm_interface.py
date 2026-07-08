from llm_sdk import Small_LLM_Model
from typing import Literal
from json import load
from .models import FunctionDefition, Types, Prompt
from .trie import Trie
from .parsing import parsing
from .states import ParameterState
import math
import re
import sys


FUNCTION = 'F'
PARAMETERS = 'P'
BASIC_REGEX = [".", "^", "$", "*", "+", "?", "|",
               "(", ")", "[", "]", "{", "}", "\\",
               r"\d", r"\D", r"\w", r"\W", r"\s",
               r"\S", r"\b", r"\B"]


class LlmInteface:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", *,
                 temperature: float = 0.3):
        infos = parsing()
        self.__functions: dict[str, FunctionDefition] = {
            f.name: f
            for f in infos["functions"]
            }
        self.__prompts: list[Prompt] = infos.get("prompts")
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
            example = ('prompt: "What is the sum of 2 and 3?n_add_numbers"\n'
                       'assistant: a=2.0,\nb=3.0\n'
                       'prompt: "Reverse the string \'hello\''
                       'fn_reverse_string\n'
                       'assistant: s=\'hello\'\n'
                       'prompt: "What is the square root of 16?'
                       'fn_get_square_root"\n'
                       'assistant: a=16.0\n'
                       'promt: "Greet john'
                       'ft_greet"\n'
                       'assistant: name=jhon\n')

            sys_prompt = ("<|im_start|>system\n"
                          "/no_think\n"
                          "Extract the parameter values "
                          "from the user prompt.\n"
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
            if isinstance(tokens, list):
                tokens.append(self.__end_key)
            else:
                tokens = [tokens, self.__end_key]
            node = root
            for token in tokens:
                if (isinstance(token, list)
                        and all(t not in node.children for t in token)):
                    node.add_children(Trie(token))
                elif token not in node.children:
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

    def _get_param_value_number(self, logits: list[float],
                                answer: str) -> None:
        dot_in_answer = answer and "." in answer.splitlines()[-1]
        tokenized_dot = [-1]
        tokenized_comma = [-1]
        if not dot_in_answer:
            dot_token = self.__model.encode(".").squeeze().tolist()
            tokenized_dot = (dot_token if isinstance(dot_token, list)
                             else [dot_token])
        if dot_in_answer:
            comma_token = self.__model.encode(",").squeeze().tolist()
            tokenized_comma = (comma_token if isinstance(comma_token, list)
                               else [comma_token])
        for index in range(len(logits)):
            if (answer.endswith("\n")
                    and index == self.__end_key):
                logits[index] = float("-inf")
                continue
            elif index == self.__end_key:
                continue
            elif not dot_in_answer and index in tokenized_dot:
                continue
            elif dot_in_answer and index in tokenized_comma:
                continue
            elif re.fullmatch(r"[\d\n]*", self.__model.decode([index])):
                continue
            logits[index] = float("-inf")

    def _get_param_value_string(self, logits: list[float],
                                answer: str, regex: Trie | None) -> None:
        for index in range(len(logits)):
            if (answer.endswith("\n")
                    and not index == self.__end_key):
                logits[index] = float("-inf")
                continue
            elif regex and index in regex.children:
                continue
            elif index == self.__end_key:
                continue
            elif not regex and re.fullmatch(r"[\w\s\\\.\,\"\']*",
                                            self.__model.decode([index])):
                continue
            logits[index] = float("-inf")

    def _get_param_value(self, tokenized_prompt: list[int],
                         param: str, param_type: Types, regex: bool) -> str:
        tokenized_param = self.__model.encode(param).squeeze().tolist()
        regex_trie = None
        if regex and param_type.type == "string":
            regex_trie = self._create_trie([self.__model.encode(
                reg
                ).squeeze().tolist() for reg in BASIC_REGEX])
        while True:
            logits = self.__model.get_logits_from_input_ids(
                tokenized_prompt + tokenized_param
            )
            if param_type.type == "number":
                self._get_param_value_number(
                    logits,
                    self.__model.decode(tokenized_param)
                )
            elif param_type.type == "string":
                self._get_param_value_string(
                    logits,
                    self.__model.decode(tokenized_param),
                    regex_trie
                    )
            logits = self._softmax(logits)
            token = logits.index(max(logits))
            if token == self.__end_key:
                break
            if regex_trie and token in regex_trie.children:
                regex_trie = regex_trie.children[token]
            tokenized_param.append(token)
            print(self.__model.decode(tokenized_param))
        return self.__model.decode(tokenized_param)

    def _generate_parameters(self, tokenized_prompt: list,
                             function_name: str) -> str:
        parameters = ""
        parameters_name = list(
            self.__functions[function_name].parameters.keys()
            )
        for param_name in parameters_name:
            parameters += param_name + "="
            parameters = self._get_param_value(
                tokenized_prompt,
                parameters,
                self.__functions[function_name].parameters[param_name],
                param_name == "regex")
        return parameters

    def _check_prompt(self, prompt: str, func: str) -> bool:
        if self.__functions[func].returns.type == "number":
            if not re.search(r"\d+\.?\d*", prompt):
                return False
        return True

    def _build_json(self, function_name: str, parameters: str):
        pass

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
        print("func:", function_answer)

        if not self._check_prompt(user_prompt, function_answer):
            raise ValueError("CADE!!!!!!!!!")

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
        parameters_answer = self._generate_parameters(
                tokenized_sys_prompt +
                tokenized_user_prompt,
                function_answer
                )
        print('params:', parameters_answer)
        #return self._build_json(function_answer, parameters_answer)

    def genrate(self) -> str:
        for p in self.__prompts:
            print(p)
            answer = self._generate_answer(f"{p!r}")
        return
