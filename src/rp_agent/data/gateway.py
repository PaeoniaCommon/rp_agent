"""DataGateway: the only code path that calls `rpdata.get_data` (SPEC.md §6.1)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import rpdata

from .store import utc_now

log = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised if a caller tries to go past the per-request hard cap (SPEC.md §6.2)."""


@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame | None
    error: BaseException | None
    retrieved_at: datetime
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.error is None


class DataGateway:
    def __init__(
        self,
        get_data: Callable[..., pd.DataFrame] | None = None,
        log_dir: Path | None = None,
        clock: Callable[[], datetime] = utc_now,
        max_calls_per_request: int = 2,
    ) -> None:
        self._get_data = get_data or rpdata.get_data
        self.log_path = (log_dir / "get_data_calls.jsonl") if log_dir else None
        self.clock = clock
        self.max_calls_per_request = max_calls_per_request
        self._lock = threading.Lock()
        self.calls = 0
        self.errors = 0
        self.empty = 0

    def fetch(
        self,
        view: str,
        params: dict[str, Any],
        *,
        user_id: str,
        thread_id: str,
        calls_already_made: int,
    ) -> FetchResult:
        if calls_already_made >= self.max_calls_per_request:
            raise BudgetExceededError(
                f"Request already made {calls_already_made} get_data calls "
                f"(max {self.max_calls_per_request})"
            )
        started = time.perf_counter()
        df: pd.DataFrame | None = None
        error: BaseException | None = None
        try:
            df = self._get_data(view, **params)
        except Exception as exc:  # noqa: BLE001 - every failure is classified by diagnose
            error = exc
        duration = time.perf_counter() - started
        retrieved_at = self.clock()

        with self._lock:
            self.calls += 1
            if error is not None:
                self.errors += 1
            elif df is not None and df.empty:
                self.empty += 1
        self._log(view, params, user_id, thread_id, df, error, duration, retrieved_at)
        return FetchResult(df=df, error=error, retrieved_at=retrieved_at, duration_s=duration)

    def _log(self, view, params, user_id, thread_id, df, error, duration, retrieved_at) -> None:
        if self.log_path is None:
            return
        if error is not None:
            outcome = "error"
        elif df is not None and df.empty:
            outcome = "empty"
        else:
            outcome = "ok"
        entry = {
            "timestamp": retrieved_at.isoformat(),
            "user_id": user_id,
            "thread_id": thread_id,
            "view": view,
            "params": params,
            "outcome": outcome,
            "error_class": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "duration_s": round(duration, 4),
            "row_count": None if df is None else len(df),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            log.exception("Could not write get_data call log")
