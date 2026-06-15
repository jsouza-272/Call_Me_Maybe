from pydantic import BaseModel, ConfigDict
from typing import Literal


class Types(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['number', 'string']


class FunctionDefition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str
