from enum import Enum


class ParameterState(Enum):
    PARAMETER = 0
    EQUAL = 1
    VALUENUMBER = 2
    VALUESTRING = 3
    END = 4
    SPACE = 5
