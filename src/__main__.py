"""Program entrypoint that generates and saves function-calling results."""

try:
    if __name__ == "__main__":
        from .llm_interface import LlmInteface

        llm = LlmInteface()
        llm.save_json(llm.genrate())

except Exception as e:
    print(f"\033[38;2;240;20;20m{e}\033[0m")

except KeyboardInterrupt:
    print("\033[38;2;240;230;20m",
          "\nEnding program",
          "\033[0m", sep="")
