"""Agent settings, read from environment variables with sensible defaults.

See SPEC.md §2 (model), §6.2 (budget), §7.3 (reuse window), §8.4 (value domains)
and §8.6 (background learning).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # LLM (SPEC.md §2, §8.6)
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    learn_model: str | None = None  # None -> same as `model`
    learn_effort: str = "low"
    max_tokens: int = 16000

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
    def resolved_learn_model(self) -> str:
        return self.learn_model or self.model

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from `RP_AGENT_*` environment variables, then apply overrides."""
        env: dict = {}
        if model := os.environ.get("RP_AGENT_MODEL"):
            env["model"] = model
        if effort := os.environ.get("RP_AGENT_EFFORT"):
            env["effort"] = effort
        if learn_model := os.environ.get("RP_AGENT_LEARN_MODEL"):
            env["learn_model"] = learn_model
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
