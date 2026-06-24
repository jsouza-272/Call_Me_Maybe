from .parsing import parsing
from llm_sdk import Small_LLM_Model
import math
import json
from sys import exit


TEMPERATURE = 0.3
TOKEN_LIMIT = 150


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    new = [math.exp((logit - max_logit) / TEMPERATURE) for logit in logits]
    logits_sum = sum(new)
    return [logit / logits_sum for logit in new]


def call(sys: list, prompt, llm: Small_LLM_Model):
    encoded = llm.encode(prompt)
    encoded_list = sys + encoded.squeeze().tolist()
    token = 0
    answer = list()
    i = 0
    while token != 151645 and i < TOKEN_LIMIT:
        logits = softmax(llm.get_logits_from_input_ids(encoded_list + answer))
        token = logits.index(max(logits))
        if token in [151667, 151668]:
            encoded_list.append(token)
        else:
            answer.append(token)
        i += 1
    if i == TOKEN_LIMIT:
        print("Token limit!", flush=True)
    return llm.decode(answer)


infos = parsing()
models = {"gpt2": "openai-community/gpt2",
          "qwen": "Qwen/Qwen3-0.6B"}
llm = Small_LLM_Model(model_name=models["qwen"])

example = ('{\n\t"prompt": "What is the sum of 2 and 3?",\n\t'
           '"name": "fn_add_numbers",\n\t"parameters": {"a": 2.0, "b": 3.0}\n}'
           '\n{\n\t"prompt": "Reverse the string \'hello\'",\n\t'
           '"name": "fn_reverse_string",\n\t"parameters": {"s": "hello"}\n}\n'
           '\n{\n\t"prompt": "Calculate the square root of 144",\n\t'
           '"name": "fn_get_square_root",\n\t"parameters": {"a": 144.0}\n}')

example2 = ('"prompt": "What is the sum of 2 and 3?"\n"function": fn_add_numbers(a=2.0, b=3.0)\n'
            '"prompt": "Reverse the string \'hello\'"\n"function": fn_reverse_string(s:"hello")\n'
            '"prompt": "Calculate the square root of 144"\n"function": fn_get_square_root(a: 144.0)')

system_prompt = ("<|im_start|>system\n"
                 "/no_think\n"
                 "You must act as a function calling system.\n"
                 "Return only a JSON object following the provided schema.\n"
                 f"schema example: {example}\n"
                 f"The functions you must use are the following: {infos.get('functions')}.\n"
                 "<|im_end|>")


sys = llm.encode(system_prompt)
sys_list = sys.squeeze().tolist()
answers = ""
with open(llm.get_path_to_vocab_file()) as file:
    vocab = {value: key for key, value in json.load(file).items()}
    print(vocab[llm.encode("").squeeze().tolist()[1]])
exit()
for p in infos.get('prompts'):
    user_prompt = f"<|im_start|>user\n {p!r} <|im_end|> <|im_start|>assistant\n"
    prompt = user_prompt
    answer = call(sys_list, prompt, llm)
    answers += answer + '\n'
    print("resposta:", answer)
print(answers)
