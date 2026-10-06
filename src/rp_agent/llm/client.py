"""Chat model construction and structured-output calls (SPEC.md §2)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


def make_chat_model(model: str, effort: str, max_tokens: int) -> Any:
    """A Claude chat model via langchain-anthropic.

    Effort is set explicitly (on claude-opus-5-5 the API default is `medium`).
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=max_tokens, output_config={"effort": effort})


def structured(llm: Any, schema: type[BaseModel], system: str, user: str) -> BaseModel:
    """Call `llm` and parse its answer into `schema`.

    Uses `method="json_schema"` (structured outputs). The default
    function-calling method forces a tool choice, which current Claude models reject.
    """
    runnable = llm.with_structured_output(schema, method="json_schema")
    result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    if isinstance(result, dict):
        result = schema.model_validate(result)
    return result
