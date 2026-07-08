from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
print(model.encode(r"\d").squeeze().tolist())