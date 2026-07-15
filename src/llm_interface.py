import math
import re
import os

from llm_sdk import Small_LLM_Model
from accelerate import Accelerator
from typing import Literal, Any
from json import load, dump

from .trie import Trie
from .parsing import parsing
from .models import FunctionDefition, Types, Prompt


FUNCTION: Literal['F'] = 'F'
PARAMETERS: Literal['P'] = 'P'
BASIC_REGEX = [".", "^", "$", "*", "+", "?", "|",
               "[", "]", r"\d", r"\D", r"\w",
               r"\W", r"\s", r"\S", r"\b", r"\B"]


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
        self.__model = Small_LLM_Model(model, device=Accelerator().device)
        if "Qwen3-0.6B" in model:
            end_key = self.__model.encode("<|im_end|>").tolist()[0][0]
        self.__end_key: int = end_key
        with open(self.__model.get_path_to_vocab_file()) as file:
            self.__vocab = {v: k for k, v in load(file).items()}
        print("="*60)
        print(f"model: {model}")
        print(f"VOCAB_SIZE:{len(self.__vocab)}")
        print("="*60)

    def _format_prompt(self, prompt: str,
                       function_answer: str = "") -> str:
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
                       'assistant: s=hello\n'
                       'prompt: "Replace all numbers in \"Hello 34 I\'m 233 '
                       'years old\" with NUMBERS.'
                       'fn_substitute_string_with_regex"\n'
                       'assistant: '
                       "source_string=Hello 34 I'm 233 years old,\n"
                       'regex=\\d+,\n'
                       'replacement=NUMBERS\n'
                       'promt: "Greet john'
                       'ft_greet"\n'
                       'assistant: name=jhon\n')

            sys_prompt = ("<|im_start|>system\n"
                          "/no_think\n"
                          "Extract the parameter values "
                          "from the user prompt.\n"
                          "Return only the paramaters of the funtion\n"
                          "Regex rules: use the shortest pattern that "
                          "matches only the target substring(s).\n"
                          "Never repeat the same group or subpattern"
                          " multiple times.\n"
                          "Never prefix/suffix the pattern with .*, .*?,"
                          " ^, $ or any surrounding context — match ONLY"
                          " the target itself.\n"
                          "One pass is always enough, like regex=\\d+.\n"
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

    def _create_trie(self, tokens_list: list[list[int]],
                     end_key: bool = False) -> Trie:
        root = Trie(-1)
        for tokens in tokens_list:
            if end_key:
                tokens.append(self.__end_key)
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
            ).tolist()[0] for func in self.__functions.values()],
            True)
        function_name: list = list()
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
            tokenized_dot = self.__model.encode(".").tolist()[0]
        if dot_in_answer:
            tokenized_comma = self.__model.encode(",").tolist()[0]
        for index in range(len(logits)):
            if (answer.endswith("\n")
                    and not index == self.__end_key):
                logits[index] = float("-inf")
                continue
            elif index == self.__end_key:
                continue
            elif index not in self.__vocab:
                logits[index] = float("-inf")
                continue
            elif not dot_in_answer and index in tokenized_dot:
                continue
            elif dot_in_answer and index in tokenized_comma:
                continue
            elif re.fullmatch(r"[\d\n]*", self.__model.decode([index])):
                continue
            logits[index] = float("-inf")

    def _get_param_value_string(self, logits: list[float],
                                answer: str, regex: bool) -> None:
        newline = not answer.splitlines()[-1].endswith("=")
        current_value = answer.splitlines()[-1].split("=")[-1]
        has_escape = bool(re.search(r"\\\w", current_value))
        has_group = current_value.count("(") >= 1
        for index in range(len(logits)):
            token = self.__model.decode([index])
            if (answer.endswith("\n")
                    and not index == self.__end_key):
                logits[index] = float("-inf")
                continue
            elif index == self.__end_key:
                continue
            elif (regex and current_value == ""
                    and re.fullmatch(r"[.^*+?]\S*", token)):
                logits[index] = float("-inf")
                continue
            elif regex and has_escape and re.fullmatch(r"\\\w", token):
                logits[index] = float("-inf")
                continue
            elif regex and has_group and "(" in token:
                logits[index] = float("-inf")
                continue
            elif newline and re.fullmatch(r"[,\n]*", token):
                continue
            elif regex and re.fullmatch(
                    r"\\\w|[\w\s+*?.\[\]()^$'-]+", token):
                continue
            elif not regex and re.fullmatch(
                r"[\w\s.,*\\\/{}\[\]():;'+=-_!?]+",
                    token):
                continue
            logits[index] = float("-inf")

    def _get_param_value_boolean(self, logits: list[float],
                                 trie: Trie) -> None:
        for index in range(len(logits)):
            if index in trie.children:
                continue
            logits[index] = float("-inf")

    def _get_param_value(self, tokenized_prompt: list[int],
                         param: str, param_type: Types, regex: bool) -> str:
        tokenized_param = self.__model.encode(param).tolist()[0]
        boolean_trie = self._create_trie([self.__model.encode(tf).tolist()[0]
                                          for tf in ["true", "false"]])
        while True:
            logits = self.__model.get_logits_from_input_ids(
                tokenized_prompt + tokenized_param
            )
            if param_type.type in ("number", "integer", "float"):
                self._get_param_value_number(
                    logits,
                    self.__model.decode(tokenized_param)
                )
            elif param_type.type == "string":
                self._get_param_value_string(
                    logits,
                    self.__model.decode(tokenized_param),
                    regex
                    )
            elif param_type.type == "boolean":
                self._get_param_value_boolean(logits, boolean_trie)
            logits = self._softmax(logits)
            token = logits.index(max(logits))
            if token == self.__end_key:
                break
            if token in boolean_trie.children:
                boolean_trie = boolean_trie.children[token]
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
            print("\nparam_name:", param_name)
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

    def _build_json(self, function_name: str,
                    parameters: str) -> dict[str, Any]:
        split_parameters = parameters.splitlines()
        formated_answer: dict[str, Any] = {"name": function_name}
        parameters_dict: dict[str, Any] = {}
        for param in split_parameters:
            param_name = param.split("=", 1)[0]
            param_value = param.split("=", 1)[1].strip().strip(",")
            param_type = self.__functions[
                function_name].parameters[param_name].type
            if param_type in ("number", "float"):
                parameters_dict.update({param_name: float(param_value)})
            elif param_type == "integer":
                parameters_dict.update({param_name: int(param_value)})
            elif param_type == "boolean":
                parameters_dict.update({param_name: bool(param_value)})
            else:
                parameters_dict.update({param_name: param_value})
        formated_answer.update({"parameters": parameters_dict})
        return formated_answer

    def _generate_answer(self, user_prompt: str) -> dict[str, Any]:
        sys_prompt = self._sys_prompt(FUNCTION)
        formated_user_prompt = self._format_prompt(user_prompt)
        tokenized_user_prompt = self.__model.encode(
            formated_user_prompt
        ).tolist()[0]
        tokenized_sys_prompt = self.__model.encode(
            sys_prompt
            ).tolist()[0]

        function_answer = self._generate_function(
            tokenized_sys_prompt + tokenized_user_prompt
            )

        if not self._check_prompt(user_prompt, function_answer):
            raise ValueError("CADE!!!!!!!!!")

        formated_user_prompt = self._format_prompt(
            user_prompt, function_answer
        )
        tokenized_user_prompt = self.__model.encode(
            formated_user_prompt
        ).tolist()[0]
        sys_prompt = self._sys_prompt(PARAMETERS)
        tokenized_sys_prompt = self.__model.encode(
                sys_prompt
                ).tolist()[0]
        parameters_answer = self._generate_parameters(
                tokenized_sys_prompt +
                tokenized_user_prompt,
                function_answer
                )
        return self._build_json(function_answer, parameters_answer)

    def genrate(self) -> list[dict[str, str]]:
        answer_list = []
        for p in self.__prompts:
            print(p)
            answer = {"prompt": f"{p!r}"}
            answer.update(self._generate_answer(f"{p!r}"))
            answer_list.append(answer)
        return answer_list

    def save_json(self, obj: list[dict[str, str]]) -> None:
        output_path = "".join(d + "/" for d in self.__output.split("/")[:-1])
        os.makedirs(output_path, exist_ok=True)
        with open(self.__output, "w", encoding="utf-8") as file:
            dump(obj, file, ensure_ascii=False, indent=4)
