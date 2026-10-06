"""Everything the graph nodes share. Built once per `RPAgent`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .catalogue import Catalogue
from .config import Settings
from .gateway import DataGateway
from .knowledge import KnowledgeBase
from .learning import Learning
from .store import DataStore
from .validation import ParamValidator
from .values import ValueDomains


@dataclass
class AgentContext:
    settings: Settings
    catalogue: Catalogue
    knowledge: KnowledgeBase
    domains: ValueDomains
    validator: ParamValidator
    store: DataStore
    gateway: DataGateway
    learning: Learning
    llm: Any
    clock: Callable[[], datetime]

    def today(self) -> str:
        return self.clock().date().isoformat()
