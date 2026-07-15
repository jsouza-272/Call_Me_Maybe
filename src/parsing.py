"""Command-line and JSON parsing helpers for input/output files."""

import sys
from typing import TypedDict
from json import load, JSONDecodeError
from pydantic import ValidationError
from .models import FunctionDefition, Prompt
from .errors import ParsingError

DEFAULT_PATHS = {'func_file': 'data/input/functions_definition.json',
                 'input_file': 'data/input/function_calling_tests.json',
                 'output_file': 'data/output/function_calling_results.json'}


class ParsingResult(TypedDict):
    """Structured result returned by the parsing pipeline."""

    output_file: str
    functions: list[FunctionDefition]
    prompts: list[Prompt]


def cli_parsing(flag: str) -> str:
    """Read the JSON path provided after a CLI flag.

    Args:
        flag: Command-line option to inspect (for example ``--input``).

    Returns:
        The JSON file path associated with the flag, or an empty string when
        the flag is missing.

    Raises:
        ParsingError: If the flag is present without a following path or with
            a non-JSON file extension.
    """
    if flag not in sys.argv:
        return ''
    index = sys.argv.index(flag)
    try:
        if not sys.argv[index + 1].endswith('.json'):
            raise ParsingError(f'Invalid extension: "{flag}" '
                               'must point to a ".json" file')
    except IndexError:
        raise ParsingError(f'Missing path after "{flag}"')
    return sys.argv[index + 1]


def parsing() -> ParsingResult:
    """Load function definitions, prompts, and output path from disk.

    Returns:
        A mapping containing the resolved output file path, parsed functions,
        and parsed prompts.

    Raises:
        ParsingError: If required files are missing, contain invalid JSON, or
            do not match the expected schema.
    """
    flags = {'func_file': cli_parsing('--functions_definition'),
             'input_file': cli_parsing('--input'),
             'output_file': cli_parsing('--output')}
    for key in flags.keys():
        if not flags[key]:
            flags[key] = DEFAULT_PATHS[key]
    try:
        with open(flags['func_file']) as file:
            try:
                functions = [FunctionDefition(**func) for func in load(file)]
            except JSONDecodeError as e:
                raise ParsingError(f"Invalid JSON in '{flags['func_file']}':"
                                   f"line {e.lineno}, column {e.colno}")
            except ValidationError as e:
                raise ParsingError("Invalid file format: "
                                   f"'{flags['func_file']}' contains invalid "
                                   "or missing fields", e)

        with open(flags['input_file']) as file:
            try:
                prompts = [Prompt(**prompt) for prompt in load(file)]
            except JSONDecodeError as e:
                raise ParsingError(f"Invalid JSON in '{flags['input_file']}':"
                                   f"line {e.lineno}, column {e.colno}")
            except ValidationError:
                raise ParsingError("Invalid file format: "
                                   f"'{flags['input_file']}' contains invalid "
                                   "or missing fields")

    except FileNotFoundError as e:
        raise ParsingError(f'File not found: {e.filename}')
    return {"output_file": flags['output_file'],
            "functions": functions,
            "prompts": prompts}
