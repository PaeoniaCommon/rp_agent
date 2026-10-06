"""Chat model construction and structured-output calls (SPEC.md §2).

The model is a `langchain_openai.ChatOpenAI`, so any OpenAI-compatible endpoint
works: set the model, API key and base URL in `Settings` (or the RP_AGENT_*
environment variables).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

STRUCTURED_METHODS = ("json_schema", "function_calling", "json_mode")


def make_chat_model(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> Any:
    """A `ChatOpenAI` for `model` at `base_url`.

    `api_key` / `base_url` left as None fall back to ChatOpenAI's own defaults
    (`OPENAI_API_KEY` / `OPENAI_BASE_URL`). `temperature` and `max_tokens` are only
    sent when set, because some models reject them.
    """
    from langchain_openai import ChatOpenAI

    if not model:
        raise ValueError("No model configured: set Settings.model or RP_AGENT_MODEL")
    kwargs: dict[str, Any] = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatOpenAI(**kwargs)


@dataclass(frozen=True)
class LLMClient:
    """A chat model plus the structured-output method to use with it."""

    chat: Any
    method: str = "json_schema"

    def __post_init__(self) -> None:
        if self.method not in STRUCTURED_METHODS:
            raise ValueError(f"structured output method must be one of {STRUCTURED_METHODS}")

    def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        """Call the model and parse its answer into `schema`."""
        runnable = self.chat.with_structured_output(schema, method=self.method)
        result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if isinstance(result, dict):
            result = schema.model_validate(result)
        return result
