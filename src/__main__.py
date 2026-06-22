from .parsing import parsing
from llm_sdk import Small_LLM_Model


def call(sys: list, prompt, llm: Small_LLM_Model):
    print('call')
    encoded = llm.encode(prompt)
    encoded_list = encoded.squeeze().tolist() + sys
    token = 0
    answer = list()
    i = 0
    print('antes do loop')
    while token != 151645 and i < 50:
        print(i)
        logits = llm.get_logits_from_input_ids(encoded_list + answer)
        token = logits.index(max(logits))
        if token in [151667, 151668]:
            encoded_list.append(token)
        else:
            answer.append(token)
        i += 1
    return llm.decode(answer)


infos = parsing()
llm = Small_LLM_Model()

example = ('{\n\t"prompt": "What is the sum of 2 and 3?",\n\t'
           '"name": "fn_add_numbers",\n\t"parameters": {"a": 2.0, "b": 3.0}\n}'
           '\n{\n\t"prompt": "Reverse the string \'hello\'",\n\t'
           '"name": "fn_reverse_string",\n\t"parameters": {"s": "hello"}\n}\n')

system_prompt = ("<|im_start|>system\n"
                 "/no_think\n"
                 "You must act as a function calling system.\n"
#                "Return only the name of the function and then parameters"
                 "Return only a JSON object following the provided schema.\n"
                 f"schema example: {example}\n"
                 f"The functions you must use are the following: {infos.get('functions')}.\n"
                 "<|im_end|>")


sys = llm.encode(system_prompt)
sys_list = sys.squeeze().tolist()
answers = ""
for p in infos.get('prompts'):
    user_prompt = f"<|im_start|>user\n {p} <|im_end|> <|im_start|>assistant\n"
    prompt = user_prompt
    answers.join(call(sys_list, prompt, llm))
    answers.join('\n')
    print(answers)
print(answers)
