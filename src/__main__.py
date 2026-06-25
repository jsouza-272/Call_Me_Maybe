from .parsing import parsing
from llm_sdk import Small_LLM_Model
from .constraining_decoding import constraining_decoding
import math
import json


TEMPERATURE = 0.3
TOKEN_LIMIT = 1000


def get_state(answer, llm: Small_LLM_Model):
    if not answer:
        return START
    decoded_answer = llm.decode(answer)
    in_brace = decoded_answer.count("{") != decoded_answer.count("}")
    in_quotes = decoded_answer.count('"') % 2 != 0
    has_name = decoded_answer.find("\"name\"") != -1
    has_func_def = any(decoded_answer.find(func.name) != -1 for func in infos['functions']) if has_name else False
    has_parameters = decoded_answer.find("\"parameters\"") != -1


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    new = [math.exp((logit - max_logit) / TEMPERATURE) for logit in logits]
    logits_sum = sum(new)
    return [logit / logits_sum for logit in new]


def call(sys: list, prompt: str, llm: Small_LLM_Model, vocab):
    encoded = llm.encode(prompt)
    encoded_list = sys + encoded.squeeze().tolist()
    token = 0
    answer = list()
    i = 0
    while token != 151645:
        logits = llm.get_logits_from_input_ids(encoded_list + answer)
        #state = get_state(answer, llm)
        logits = softmax(logits)
        #logits = constraining_decoding(state, vocab, logits, llm)
        token = logits.index(max(logits))
        if token in [151667, 151668]:
            encoded_list.append(token)
        else:
            answer.append(token)
        i += 1
    if i == TOKEN_LIMIT:
        print("Token limit!", flush=True)
        return "fail!\n"
    return llm.decode(answer)


infos = parsing()
models = {"gpt2": "openai-community/gpt2",
          "qwen": "Qwen/Qwen3-0.6B"}
llm = Small_LLM_Model(model_name=models["qwen"])
print(f"device: {llm._device}\n")

example = ('{\n\t"name": "fn_add_numbers",\n\t"parameters": {"a": 2.0, "b": 3.0}\n}'
           '\n{\n\t"name": "fn_reverse_string",\n\t"parameters": {"s": "hello"}\n}\n'
           '\n{\n\tname": "fn_get_square_root",\n\t"parameters": {"a": 144.0}\n}')

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
#with open(llm.get_path_to_vocab_file()) as file:
#    vocab = {value: key for key, value in json.load(file).items()}
#    print(vocab[llm.encode("").squeeze().tolist()[1]])
#exit()
with open(llm.get_path_to_vocab_file()) as file:
    vocab = {value: key for key, value in
            json.load(file).items()}
#for p in infos.get('prompts'):
user_prompt = f"<|im_start|>user\n Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS <|im_end|> <|im_start|>assistant\n"
    #prompt = user_prompt + '{\n\t"prompt":' + f'{p!r},\n\t"name":'
    #print("prompt:", p)
answer = call(sys_list, user_prompt , llm, vocab)
answers += answer + '\n'
print("resposta:", answer, end="\n\n")
print(answers)
