"""Pydantic models used by the function-calling pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Types(BaseModel):
    """Type descriptor for parameters and return values."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean", "integer", "float"]

    def __repr__(self) -> str:
        return self.type


class FunctionDefinition(BaseModel):
    """Function definition loaded from the input schema file."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types

    def __repr__(self) -> str:
        params = ", ".join(f"{name}: {param_type!r}"
                           for name, param_type in self.parameters.items())
        return (f"{self.name}({params}) -> {self.returns!r}\n"
                f"description: {self.description}")


class Prompt(BaseModel):
    """Prompt model loaded from input prompts JSON."""

    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallOutput(BaseModel):
    """Output item written to the output file."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]


# Backward-compatible alias for existing imports.
FunctionDefition = FunctionDefinition
