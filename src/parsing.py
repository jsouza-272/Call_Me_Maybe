"""CLI and JSON parsing helpers."""

from argparse import ArgumentParser
from json import JSONDecodeError, load
from typing import Any, TypedDict

from pydantic import ValidationError

from .errors import ParsingError
from .models import FunctionDefinition, Prompt

DEFAULT_PATHS = {
    "functions_definition": "data/input/functions_definition.json",
    "input": "data/input/function_calling_tests.json",
    "output": "data/output/function_calling_results.json",
}


class ParsingResult(TypedDict):
    """Parsed input files and output path."""

    output_file: str
    functions: list[FunctionDefinition]
    prompts: list[Prompt]


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m src")
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_PATHS["functions_definition"],
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_PATHS["input"],
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PATHS["output"],
        help="Path to the output JSON file.",
    )
    return parser


def _validate_json_path(path: str, flag: str) -> None:
    if not path.endswith(".json"):
        raise ParsingError(
            f'Invalid extension: "{flag}" must point to a ".json" file'
        )


def _load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as file:
            return load(file)
    except FileNotFoundError as error:
        raise ParsingError(f"File not found: {error.filename}") from error
    except JSONDecodeError as error:
        raise ParsingError(
            f"Invalid JSON in '{path}': line {error.lineno}, "
            f"column {error.colno}"
        ) from error


def parsing() -> ParsingResult:
    """Parse CLI args and load validated project inputs."""

    args = _build_parser().parse_args()

    _validate_json_path(args.functions_definition, "--functions_definition")
    _validate_json_path(args.input, "--input")
    _validate_json_path(args.output, "--output")

    functions_json = _load_json(args.functions_definition)
    prompts_json = _load_json(args.input)

    try:
        functions = [FunctionDefinition(**item) for item in functions_json]
    except ValidationError as error:
        raise ParsingError(
            "Invalid file format: "
            f"'{args.functions_definition}' contains invalid or missing fields"
        ) from error

    try:
        prompts = [Prompt(**item) for item in prompts_json]
    except ValidationError as error:
        raise ParsingError(
            "Invalid file format: "
            f"'{args.input}' contains invalid or missing fields"
        ) from error

    return {
        "output_file": args.output,
        "functions": functions,
        "prompts": prompts,
    }
