from pydantic import BaseModel, ConfigDict
from typing import Literal


class Types(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['number', 'string', 'boolean', 'integer', 'float']

    def __repr__(self) -> Literal[
         'number', 'string', 'boolean', 'integer', 'float']:
        return self.type


class FunctionDefition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types

    def __repr__(self) -> str:
        params = "".join(f"{k}: {v!r}, "
                         for k, v in self.parameters.items())[:-2]
        rep = (f"{self.name}({params}) -> {self.returns!r}\n"
               f"description: {self.description}")
        return rep


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str

    def __repr__(self) -> str:
        return self.prompt
