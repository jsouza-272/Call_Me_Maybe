from llm_sdk import Small_LLM_Model
from .models import FunctionDefition
from .trie import Trie


def create_trie(functions: list[FunctionDefition], model: Small_LLM_Model) -> Trie:
    root = Trie(-1)
    for func in functions:
        tokens = model.encode(func.name).squeeze().tolist()
        node = root
        for token in tokens:
            if token not in node.children:
                node.add_children(Trie(token))
            node = node.children[token]
        node.is_end = True
    return root


def get_function_name(vocab, logits, trie: Trie):
    for index in range(len(logits)):
        if vocab[index] not in trie.children:
            logits[index] = float("-inf")
        


def constraining_decoding(vocab, logits, model: Small_LLM_Model, avalible_functions: list[FunctionDefition]):
    pass
