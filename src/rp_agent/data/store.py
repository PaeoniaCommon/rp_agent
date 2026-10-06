"""Per-user in-memory data store (SPEC.md §7)."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import pandas as pd

from .canonical import CanonicalRequest


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    user_id: str
    view: str
    params: dict[str, Any]
    retrieved_at: datetime
    row_count: int
    columns: tuple[str, ...]
    thread_id: str
    original_request: str
    df: pd.DataFrame = field(repr=False, compare=False)
    cache_key: str = field(default="", repr=False)

    def metadata(self) -> dict[str, Any]:
        """Everything except the DataFrame. This is all the LLM is ever shown (SPEC.md §3)."""
        return {
            "dataset_id": self.dataset_id,
            "view": self.view,
            "params": dict(self.params),
            "retrieved_at": self.retrieved_at.isoformat().replace("+00:00", "Z"),
            "row_count": self.row_count,
            "columns": list(self.columns),
        }


class DataStore(Protocol):
    def put(
        self,
        user_id: str,
        df: pd.DataFrame,
        request: CanonicalRequest,
        retrieved_at: datetime,
        original_request: str,
        thread_id: str = "",
        cache_key: str = "",
    ) -> DatasetRecord: ...

    def get(self, user_id: str, dataset_id: str) -> DatasetRecord: ...

    def get_df(self, user_id: str, dataset_id: str) -> pd.DataFrame: ...

    def find_recent(
        self, user_id: str, cache_key: str, within: timedelta, now: datetime
    ) -> DatasetRecord | None: ...

    def list(self, user_id: str) -> list[DatasetRecord]: ...


class InMemoryDataStore:
    """A dict keyed by user_id. Users never see each other's records."""

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self.clock = clock
        self._records: dict[str, dict[str, DatasetRecord]] = {}
        self._lock = threading.Lock()

    def _new_id(self) -> str:
        while True:
            dataset_id = f"ds_{secrets.token_hex(4)}"
            if not any(dataset_id in recs for recs in self._records.values()):
                return dataset_id

    def put(
        self,
        user_id: str,
        df: pd.DataFrame,
        request: CanonicalRequest,
        retrieved_at: datetime,
        original_request: str,
        thread_id: str = "",
        cache_key: str = "",
    ) -> DatasetRecord:
        with self._lock:
            record = DatasetRecord(
                dataset_id=self._new_id(),
                user_id=user_id,
                view=request.view,
                params=request.params_dict,
                retrieved_at=retrieved_at,
                row_count=len(df),
                columns=tuple(df.columns),
                thread_id=thread_id,
                original_request=original_request,
                df=df.copy(),
                cache_key=cache_key or request.key(),
            )
            self._records.setdefault(user_id, {})[record.dataset_id] = record
            return self._copy(record)

    @staticmethod
    def _copy(record: DatasetRecord) -> DatasetRecord:
        # Callers get a copy, so the stored frame cannot be changed.
        return replace(record, df=record.df.copy(), params=dict(record.params))

    def get(self, user_id: str, dataset_id: str) -> DatasetRecord:
        with self._lock:
            try:
                return self._copy(self._records[user_id][dataset_id])
            except KeyError:
                raise KeyError(f"No dataset {dataset_id!r} for user {user_id!r}") from None

    def get_df(self, user_id: str, dataset_id: str) -> pd.DataFrame:
        return self.get(user_id, dataset_id).df

    def find_recent(
        self, user_id: str, cache_key: str, within: timedelta, now: datetime
    ) -> DatasetRecord | None:
        """The newest record for this user and key retrieved within `within` of `now`."""
        with self._lock:
            matches = [
                r
                for r in self._records.get(user_id, {}).values()
                if r.cache_key == cache_key and now - r.retrieved_at <= within
            ]
            if not matches:
                return None
            return self._copy(max(matches, key=lambda r: r.retrieved_at))

    def list(self, user_id: str) -> list[DatasetRecord]:
        with self._lock:
            records = sorted(
                self._records.get(user_id, {}).values(), key=lambda r: r.retrieved_at
            )
            return [self._copy(r) for r in records]


def cache_key_for(view: str, params: Mapping[str, Any], defaults: Mapping[str, Any]) -> str:
    return CanonicalRequest.build(view, params).key(defaults)
