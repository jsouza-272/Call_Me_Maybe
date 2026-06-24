from llm_sdk import Small_LLM_Model


def get_allowed(state):
    pass


def constraining_decoding(state, vocab, logits, model: Small_LLM_Model):
    allowed = model.encode(get_allowed(state))
    for 
