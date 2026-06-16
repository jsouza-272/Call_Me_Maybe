from .parsing import parsing
from json import load
from llm_sdk import Small_LLM_Model


flags = parsing()
llm = Small_LLM_Model()

prompt = "<| "
