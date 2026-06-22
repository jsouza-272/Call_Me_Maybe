import sys
from json import load, JSONDecodeError
from pydantic import ValidationError
from typing import Any
from .models import FunctionDefition, Prompt
from .errors import ParsingError

DEFAULT_PATHS = {'func_file': 'data/input/functions_definition.json',
                 'input_file': 'data/input/function_calling_tests.json',
                 'output_file': 'data/output/function_calling_results.json'}


def cli_parsing(flag: str) -> str:
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


def parsing() -> dict[str, Any]:
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
            except ValidationError:
                raise ParsingError("Invalid file format: "
                                   f"'{flags['func_file']}' contains invalid "
                                   "or missing fields")

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
