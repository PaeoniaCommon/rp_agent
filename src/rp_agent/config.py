"""Agent settings, read from environment variables with sensible defaults.

See SPEC.md §2 (model), §6.2 (budget), §7.3 (reuse window), §8.4 (value domains)
and §8.6 (background learning).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    # LLM via langchain_openai.ChatOpenAI (SPEC.md §2). Any OpenAI-compatible endpoint.
    model: str = ""
    api_key: str | None = field(default=None, repr=False)  # None -> OPENAI_API_KEY
    base_url: str | None = None  # None -> OPENAI_BASE_URL, else the OpenAI API
    temperature: float | None = None  # only sent when set
    max_tokens: int | None = None  # only sent when set
    timeout: float | None = None
    # How structured output is requested: "json_schema" (default), or
    # "function_calling" / "json_mode" for endpoints without JSON-schema support.
    structured_output_method: str = "json_schema"

    # Background learning model (SPEC.md §8.6). None -> same as the main model.
    learn_model: str | None = None
    learn_api_key: str | None = field(default=None, repr=False)
    learn_base_url: str | None = None

    # Calling get_data (SPEC.md §6.2)
    max_auto_calls_per_request: int = 1
    max_calls_per_request: int = 2

    # Reuse window (SPEC.md §7.3)
    reuse_window: timedelta = timedelta(minutes=30)

    # Paths (SPEC.md §6.1, §8.2)
    knowledge_dir: Path = Path("knowledge")
    log_dir: Path = Path("logs")

    # Value domains (SPEC.md §8.4)
    small_list_max: int = 30
    max_indexed_values: int = 10_000
    prompt_shortlist_size: int = 15
    review_options_size: int = 10
    max_aliases: int = 500
    param_domains: dict[str, str] = field(default_factory=dict)

    # Knowledge size limits (SPEC.md §8.2)
    max_rules_per_file: int = 25
    max_examples_per_view: int = 10

    # Background learning (SPEC.md §8.6)
    learning_queue_size: int = 1000
    learning_shutdown_timeout: float = 10.0

    @property
    def learn_uses_main_model(self) -> bool:
        return (self.learn_model in (None, self.model)
                and self.learn_api_key in (None, self.api_key)
                and self.learn_base_url in (None, self.base_url))

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from `RP_AGENT_*` environment variables, then apply overrides."""
        env: dict = {}
        for name, key in (
            ("RP_AGENT_MODEL", "model"),
            ("RP_AGENT_API_KEY", "api_key"),
            ("RP_AGENT_BASE_URL", "base_url"),
            ("RP_AGENT_STRUCTURED_OUTPUT", "structured_output_method"),
            ("RP_AGENT_LEARN_MODEL", "learn_model"),
            ("RP_AGENT_LEARN_API_KEY", "learn_api_key"),
            ("RP_AGENT_LEARN_BASE_URL", "learn_base_url"),
        ):
            if value := os.environ.get(name):
                env[key] = value
        if (temperature := _env_float("RP_AGENT_TEMPERATURE")) is not None:
            env["temperature"] = temperature
        if max_tokens := os.environ.get("RP_AGENT_MAX_TOKENS"):
            env["max_tokens"] = int(max_tokens)
        if (timeout := _env_float("RP_AGENT_TIMEOUT")) is not None:
            env["timeout"] = timeout
        if knowledge_dir := os.environ.get("RP_AGENT_KNOWLEDGE_DIR"):
            env["knowledge_dir"] = Path(knowledge_dir)
        if log_dir := os.environ.get("RP_AGENT_LOG_DIR"):
            env["log_dir"] = Path(log_dir)
        if domains := os.environ.get("RP_AGENT_PARAM_DOMAINS"):
            # e.g. "node=closed_unpublished,limit_id=open"
            env["param_domains"] = dict(
                item.split("=", 1) for item in domains.split(",") if "=" in item
            )
        env["learning_queue_size"] = _env_int("RP_AGENT_LEARNING_QUEUE_SIZE", 1000)
        env.update(overrides)
        return cls(**env)
