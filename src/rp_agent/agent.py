"""RPAgent: the public entry point (SPEC.md §9.1)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from langgraph.types import Command

from .config import Settings
from .data.catalogue import Catalogue
from .data.gateway import DataGateway
from .data.store import DataStore, InMemoryDataStore, utc_now
from .data.validation import ParamValidator
from .graph import build_graph
from .graph.context import AgentContext
from .graph.state import initial_state
from .llm import LLMClient, make_chat_model
from .memory.knowledge import KnowledgeBase
from .memory.learning import Learning, LearningWorker
from .memory.values import ValueDomains

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reply:
    text: str
    dataset: dict[str, Any] | None = None  # metadata only; use agent.store for the DataFrame
    review: dict[str, Any] | None = None  # set while waiting for human review
    outcome: str = ""

    @property
    def waiting_for_review(self) -> bool:
        return self.review is not None


class RPAgent:
    """Turns one data request into at most one economical `rpdata.get_data` call."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: Any | None = None,
        learn_llm: Any | None = None,
        store: DataStore | None = None,
        get_data: Callable[..., pd.DataFrame] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.settings = settings or Settings.from_env()
        s = self.settings
        if llm is None:
            llm = make_chat_model(s.model, s.api_key, s.base_url, s.temperature,
                                  s.max_tokens, s.timeout)
        if learn_llm is None:
            learn_llm = llm if s.learn_uses_main_model else make_chat_model(
                s.learn_model or s.model, s.learn_api_key or s.api_key,
                s.learn_base_url or s.base_url, s.temperature, s.max_tokens, s.timeout)
        method = s.structured_output_method
        catalogue = Catalogue()
        knowledge = KnowledgeBase(s.knowledge_dir, clock, s.max_rules_per_file,
                                  s.max_examples_per_view, s.max_aliases)
        domains = ValueDomains(catalogue, knowledge, s, clock)
        self.learning = Learning(
            LearningWorker(knowledge, domains, catalogue, LLMClient(learn_llm, method)),
            queue_size=s.learning_queue_size,
            shutdown_timeout=s.learning_shutdown_timeout,
        )
        self.store: DataStore = store or InMemoryDataStore(clock)
        self.gateway = DataGateway(get_data, s.log_dir, clock, s.max_calls_per_request)
        self.ctx = AgentContext(
            settings=s, catalogue=catalogue, knowledge=knowledge, domains=domains,
            validator=ParamValidator(catalogue, knowledge, domains, s),
            store=self.store, gateway=self.gateway, learning=self.learning, llm=LLMClient(llm, method),
            clock=clock,
        )
        self.graph = build_graph(self.ctx)

    @property
    def knowledge(self) -> KnowledgeBase:
        return self.ctx.knowledge

    def chat(self, message: Any, config: dict[str, Any], *, refresh: bool = False) -> Reply:
        """Handle a new request, or the answer to a pending review in this thread.

        `config["configurable"]` must hold `user_id` and `thread_id`. Returns as soon as
        the reply is ready; it never waits for learning (SPEC.md §8.6).
        """
        configurable = (config or {}).get("configurable", {})
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")
        if not user_id:
            raise ValueError("config['configurable']['user_id'] is required")
        if not thread_id:
            raise ValueError("config['configurable']['thread_id'] is required")
        graph_config = {"configurable": {"thread_id": f"{user_id}::{thread_id}"}}

        if self.pending_review(config) is not None:
            result = self.graph.invoke(Command(resume=message), graph_config)
        else:
            text = str(message)
            refresh = refresh or _asks_for_refresh(text)
            result = self.graph.invoke(initial_state(text, user_id, thread_id, refresh),
                                       graph_config)
        return self._reply(result, graph_config)

    def pending_review(self, config: dict[str, Any]) -> dict[str, Any] | None:
        configurable = (config or {}).get("configurable", {})
        graph_config = {"configurable": {
            "thread_id": f"{configurable.get('user_id')}::{configurable.get('thread_id')}"}}
        snapshot = self.graph.get_state(graph_config)
        for task in snapshot.tasks or ():
            for intr in task.interrupts or ():
                return intr.value
        return None

    def _reply(self, result: dict[str, Any], graph_config: dict[str, Any]) -> Reply:
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            payload = interrupts[0].value
            return Reply(text=payload["text"], review=payload, outcome="review")
        return Reply(text=result.get("reply", ""), dataset=result.get("dataset"),
                     outcome=result.get("outcome", ""))

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for background learning to finish (tests, scripts, shutdown)."""
        return self.learning.flush(timeout)

    def close(self) -> None:
        """Drain learning and stop the worker."""
        self.learning.close()


_REFRESH_WORDS = ("refresh", "re-pull", "repull", "fresh data", "re-run", "rerun")


def _asks_for_refresh(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in _REFRESH_WORDS)
