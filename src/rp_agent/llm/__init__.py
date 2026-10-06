"""The language model: client, prompts and structured-output schemas."""

from .client import LLMClient, make_chat_model

__all__ = ["LLMClient", "make_chat_model"]
