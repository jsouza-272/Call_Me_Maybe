import sys
import json
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
        elif sys.argv[index + 1].count('.json') > 1:
            raise ParsingError(f'Invalid extension: "{flag}" must point to '
                               'a file with exactly one ".json" extension')
    except IndexError:
        raise ParsingError(f'Missing path after "{flag}"')
    return sys.argv[index + 1]


def parsing():
    flags = {'func_file': cli_parsing('--functions_definition'),
             'input_file': cli_parsing('--input'),
             'output_file': cli_parsing('--output')}
    for key in flags.keys():
        if not flags[key]:
            flags[key] = DEFAULT_PATHS[key]
    try:
        with open(flags['func_file']) as file:
            for func in json.load(file):
                FunctionDefition(**func)
        with open(flags['input_file']) as file:
            for prompt in json.load(file):
                Prompt(**prompt)
    except FileNotFoundError as e:
        raise ParsingError(f'File not found: {e.filename}')
