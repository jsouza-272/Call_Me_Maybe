"""Pydantic models used to validate project input files."""

from pydantic import BaseModel, ConfigDict
from typing import Literal


class Types(BaseModel):
    """Represent a supported primitive type in function schemas."""

    model_config = ConfigDict(extra='forbid')
    type: Literal['number', 'string', 'boolean', 'integer', 'float']

    def __repr__(self) -> Literal[
         'number', 'string', 'boolean', 'integer', 'float']:
        """Return the raw type name for prompt rendering.

        Returns:
            The type value stored in the model.
        """
        return self.type


class FunctionDefition(BaseModel):
    """Describe one callable function and its signature metadata."""

    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types

    def __repr__(self) -> str:
        """Render the function schema in a compact prompt-friendly string.

        Returns:
            Formatted function signature plus description.
        """
        params = "".join(f"{k}: {v!r}, "
                         for k, v in self.parameters.items())[:-2]
        rep = (f"{self.name}({params}) -> {self.returns!r}\n"
               f"description: {self.description}")
        return rep


class Prompt(BaseModel):
    """Represent a single user prompt loaded from input JSON."""

    model_config = ConfigDict(extra='forbid')
    prompt: str

    def __repr__(self) -> str:
        """Return the raw prompt text.

        Returns:
            Prompt string contained in this model.
        """
        return self.prompt
