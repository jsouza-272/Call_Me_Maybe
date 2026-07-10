from llm_sdk import Small_LLM_Model

print(Small_LLM_Model().encode("\\\\").squeeze().tolist())
print(Small_LLM_Model().encode(r"\d+").squeeze().tolist())
