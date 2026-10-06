"""rp_agent: a LangGraph agent that retrieves data from rpdata economically.

See SPEC.md for the full specification and README.md for usage.
"""

from __future__ import annotations

from .agent import Reply, RPAgent
from .config import Settings
from .data.store import DatasetRecord, InMemoryDataStore

__version__ = "0.1.0"

__all__ = ["DatasetRecord", "InMemoryDataStore", "RPAgent", "Reply", "Settings"]
