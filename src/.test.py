from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
print(model.encode("fn_add_numbers").squeeze().tolist())
print(model.encode("fn_greet").squeeze().tolist())
print(model.encode("fn_get_square_root").squeeze().tolist())
print(model.encode("fn_get_square_root\nfn_greet").squeeze().tolist())