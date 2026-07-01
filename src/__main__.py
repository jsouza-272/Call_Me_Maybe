from .llm_interface import LlmInteface
from llm_sdk import Small_LLM_Model


TEMPERATURE = 0.3
TOKEN_LIMIT = 1000

try:
    if __name__ == "__main__":
        llm = LlmInteface()
        llm.genrate()

        # ===========================
        #llm = Small_LLM_Model()
        #print(llm.encode("<|im_start|>"))
        #print(llm.encode("<|im_end|>"))
        #print(llm.encode("<think>"))
        #print(llm.encode("</think>"))
        #print("coisa", llm.decode([151644]), "coisa", sep="")
except KeyboardInterrupt:
    print("\n\nPARA, PARA, PARA!!!!!!!!!!\n\n")