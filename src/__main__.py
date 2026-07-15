"""Program entry point for function-calling generation."""

from .errors import ParsingError
from .llm_interface import LlmInterface


def main() -> int:
    """Run the end-to-end generation pipeline."""

    try:
        interface = LlmInterface()
        interface.save_json(interface.generate())
    except KeyboardInterrupt:
        print("Execution interrupted by user.")
        return 130
    except (ParsingError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1
    except Exception as error:  # defensive fallback for graceful behavior
        print(f"Unexpected error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
