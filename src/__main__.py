from .parsing import parsing
from json import load
from llm_sdk import Small_LLM_Model


flags = parsing()
llm = Small_LLM_Model()

functions = {}
with open(flags['func_file']) as file:
    functions = load(file)

exemple = ('[\n{\n"prompt": "What is the sum of 2 and 3?",\n'
           '"name": "fn_add_numbers",\n"parameters": {"a": 2.0, "b": 3.0}\n},'
           '{\n"prompt": "Reverse the string \'hello\'",\n'
           '"name": "fn_reverse_string",\n"parameters": {"s": "hello"}\n}\n]')

system_prompt = ("<|im_start|>system"
                 " Voce deve agir como tool/function calling."
                 "As funcoes que vc deve usar estao "
                 f"em formato json e sao as seguintes: {functions}."
                 "a resposta deve estar em json aqui vao algums "
                 f"exemplos de respostas esperada: {exemple}.\n"
                 "<|im_end|>" )

prompt =  \
<|im_start|>user \
[Pergunta do utilizador] \
<|im_end|> \
<|im_start|>assistant"
