"""Canonical requests and cache keys (SPEC.md §7.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _canonical_value(value: Any) -> Any:
    # 85 and 85.0 are the same request; bools are never treated as numbers.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


@dataclass(frozen=True)
class CanonicalRequest:
    """A view plus the exact parameters sent to get_data (`None`s dropped, keys sorted)."""

    view: str
    params: tuple[tuple[str, Any], ...]

    @classmethod
    def build(cls, view: str, params: Mapping[str, Any]) -> CanonicalRequest:
        clean = {k: v for k, v in params.items() if v is not None}
        return cls(view=view, params=tuple(sorted(clean.items())))

    @property
    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def key(self, defaults: Mapping[str, Any] | None = None) -> str:
        """Stable cache key. A param equal to a `certain` learned default (an equivalence
        note, SPEC.md §8.1) is left out, so "omitted" and "default" match."""
        defaults = defaults or {}
        keyed = {
            k: _canonical_value(v)
            for k, v in self.params
            if not (k in defaults and _canonical_value(defaults[k]) == _canonical_value(v))
        }
        return json.dumps({"view": self.view, "params": keyed}, sort_keys=True, default=str)
