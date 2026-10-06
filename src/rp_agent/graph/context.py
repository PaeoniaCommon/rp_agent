"""Everything the graph nodes share. Built once per `RPAgent`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..config import Settings
from ..data.catalogue import Catalogue
from ..data.gateway import DataGateway
from ..data.store import DataStore
from ..data.validation import ParamValidator
from ..memory.knowledge import KnowledgeBase
from ..memory.learning import Learning
from ..memory.values import ValueDomains


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
