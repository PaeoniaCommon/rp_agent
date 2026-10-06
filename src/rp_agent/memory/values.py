"""Parameter value domains, the value index and shortlists (SPEC.md §8.4).

Short published lists (e.g. `limit_level`) are read live from rpdata and shown
in full. Long lists (e.g. `node`) are kept in a capped index outside the notes,
and only a shortlist of likely matches is ever shown to the LLM or a reviewer.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

SMALL_CLOSED = "small_closed"
LARGE_CLOSED = "large_closed"
CLOSED_UNPUBLISHED = "closed_unpublished"
OPEN = "open"
DOMAINS = (SMALL_CLOSED, LARGE_CLOSED, CLOSED_UNPUBLISHED, OPEN)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def squash(value: str) -> str:
    """Case- and whitespace-insensitive form used for normalisation (§5.5)."""
    return re.sub(r"\s+", "", value).upper()


class ValueIndex:
    """Known valid values for one parameter, persisted as JSON lines.

    Advisory only: a value missing from the index is still checked with
    `Parameter.validate`, so eviction can at worst cause a review.
    """

    def __init__(self, path: Path, cap: int, clock: Callable[[], datetime]) -> None:
        self.path = path
        self.cap = cap
        self.clock = clock
        self._lock = threading.RLock()
        self._values: dict[str, dict[str, Any]] = {}
        self._by_squash: dict[str, set[str]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._insert(entry["value"], entry)

    def _insert(self, value: str, entry: dict[str, Any]) -> None:
        self._values[value] = entry
        self._by_squash.setdefault(squash(value), set()).add(value)

    def _remove(self, value: str) -> None:
        self._values.pop(value, None)
        bucket = self._by_squash.get(squash(value))
        if bucket:
            bucket.discard(value)
            if not bucket:
                del self._by_squash[squash(value)]

    def __contains__(self, value: object) -> bool:
        with self._lock:
            return value in self._values

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def values(self) -> list[str]:
        with self._lock:
            return list(self._values)

    def squash_matches(self, value: str) -> list[str]:
        with self._lock:
            return sorted(self._by_squash.get(squash(value), ()))

    def most_used(self, n: int) -> list[str]:
        with self._lock:
            ranked = sorted(
                self._values.values(), key=lambda e: (-e.get("uses", 0), e["value"])
            )
            return [e["value"] for e in ranked[:n]]

    def add(self, value: str, used: bool = True) -> None:
        with self._lock:
            today = self.clock().date().isoformat()
            entry = self._values.get(value)
            if entry is None:
                entry = {"value": value, "first_seen": today, "last_seen": today, "uses": 0}
                self._insert(value, entry)
            entry["last_seen"] = today
            if used:
                entry["uses"] = entry.get("uses", 0) + 1
            if len(self._values) > self.cap:
                ranked = sorted(self._values.values(),
                                key=lambda e: (e.get("last_seen", ""), e.get("uses", 0)))
                for victim in ranked[: len(self._values) - self.cap]:
                    self._remove(victim["value"])

    def discard(self, value: str) -> None:
        with self._lock:
            self._remove(value)

    def save(self) -> None:
        with self._lock:
            lines = [json.dumps(e, sort_keys=True) for e in self._values.values()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class ValueDomains:
    """Classifies parameters and owns the value indexes (one per long-list parameter)."""

    def __init__(self, catalogue, knowledge, settings, clock: Callable[[], datetime]) -> None:
        self.catalogue = catalogue
        self.knowledge = knowledge
        self.settings = settings
        self.clock = clock
        self._indexes: dict[str, ValueIndex] = {}
        self._lock = threading.Lock()

    def domain(self, param: str) -> str:
        if param in self.settings.param_domains:
            return self.settings.param_domains[param]
        allowed = self.catalogue.params[param].allowed_values
        if allowed:
            return SMALL_CLOSED if len(allowed) <= self.settings.small_list_max else LARGE_CLOSED
        learned = self.knowledge.value_domain(param)
        return learned if learned in DOMAINS else OPEN

    def allowed_values(self, param: str) -> list[str]:
        return list(self.catalogue.params[param].allowed_values or ())

    def index(self, param: str) -> ValueIndex:
        with self._lock:
            if param not in self._indexes:
                self._indexes[param] = ValueIndex(
                    self.knowledge.root / "values" / f"{param}.jsonl",
                    cap=self.settings.max_indexed_values,
                    clock=self.clock,
                )
            return self._indexes[param]

    def known_values(self, param: str) -> list[str]:
        """Every value the agent may normalise against (§5.5)."""
        domain = self.domain(param)
        if domain in (SMALL_CLOSED, LARGE_CLOSED):
            return self.allowed_values(param)
        if domain == CLOSED_UNPUBLISHED:
            return self.index(param).values()
        return []

    def normalisation_candidates(self, param: str, value: Any) -> list[str]:
        """Known values equal to `value` ignoring case and whitespace."""
        if not isinstance(value, str):
            return []
        domain = self.domain(param)
        if domain in (SMALL_CLOSED, LARGE_CLOSED):
            return sorted({v for v in self.allowed_values(param) if squash(v) == squash(value)})
        if domain == CLOSED_UNPUBLISHED:
            return self.index(param).squash_matches(value)
        return []

    def shortlist(self, param: str, request_text: str, k: int,
                  extra_terms: tuple[str, ...] = ()) -> list[str]:
        """Up to `k` likely values for a long-list parameter, picked deterministically."""
        values = self.known_values(param)
        if not values:
            return []
        by_squash: dict[str, list[str]] = {}
        for v in values:
            by_squash.setdefault(squash(v), []).append(v)

        picked: list[str] = []

        def take(candidates) -> None:
            for c in candidates:
                if c not in picked and len(picked) < k:
                    picked.append(c)

        # 1. alias phrases that appear in the request
        take(a["value"] for a in self.knowledge.aliases_in_text(param, request_text)
             if a["value"] in set(values))
        words = [w for w in _WORD_RE.findall(request_text) if len(w) >= 2]
        words += list(extra_terms)
        # 2. exact / case- and whitespace-insensitive matches
        for w in words:
            take(by_squash.get(squash(w), []))
        # 3. prefix and token matches ("equity" -> EQUITIES, EQ_EMEA, ...)
        for w in words:
            sw = squash(w)
            if len(sw) < 2:
                continue
            take(sorted(v for v in values if squash(v).startswith(sw)
                        or any(t.startswith(sw) for t in squash(v).split("_"))
                        or (len(sw) >= 4 and sw.startswith(squash(v).split("_")[0])
                            and len(squash(v).split("_")[0]) >= 2)))
        # 4. close fuzzy matches
        upper_values = {squash(v): v for v in values}
        for w in words:
            for m in difflib.get_close_matches(squash(w), list(upper_values), n=k, cutoff=0.8):
                take([upper_values[m]])
        # 5. pad with the most used values
        if len(picked) < k:
            take(self.most_used(param, k))
        return picked[:k]

    def most_used(self, param: str, n: int) -> list[str]:
        domain = self.domain(param)
        if domain == CLOSED_UNPUBLISHED:
            return self.index(param).most_used(n)
        return self.allowed_values(param)[:n]
