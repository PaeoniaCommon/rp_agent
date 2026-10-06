"""The rpdata catalogue: views and parameter metadata, built once at start-up.

Only the public rpdata API is used (SPEC.md §4). `Parameter.validate` is the
free, per-parameter pre-flight check (SPEC.md §14, decision 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rpdata


@dataclass(frozen=True)
class ViewInfo:
    name: str
    description: str
    mandatory_params: tuple[str, ...]
    optional_params: tuple[str, ...]
    columns: tuple[str, ...]

    @property
    def all_params(self) -> tuple[str, ...]:
        return self.mandatory_params + self.optional_params


class Catalogue:
    """Cached view and parameter metadata from rpdata."""

    def __init__(self) -> None:
        self.views: dict[str, ViewInfo] = {}
        for name in rpdata.list_views():
            v = rpdata.get_view(name)
            self.views[name] = ViewInfo(
                name=v.name,
                description=v.description,
                mandatory_params=tuple(v.mandatory_params),
                optional_params=tuple(v.optional_params),
                columns=tuple(v.columns),
            )
        self.params: dict[str, rpdata.Parameter] = {p.name: p for p in rpdata.list_params()}

    @property
    def view_names(self) -> list[str]:
        return list(self.views)

    def validate_param(self, view: str, name: str, value: Any) -> Any:
        """Run rpdata's own validator for one parameter. Raises an RPDataError subclass."""
        return self.params[name].validate(view, value)

    @staticmethod
    def native_validate_params():
        """`rpdata.validate_params` if it exists (proposal R1), found by feature detection."""
        return getattr(rpdata, "validate_params", None)

    def describe_views(self) -> str:
        lines = []
        for v in self.views.values():
            lines.append(
                f"- {v.name}: {v.description}\n"
                f"    mandatory: {', '.join(v.mandatory_params) or '(none)'}\n"
                f"    optional: {', '.join(v.optional_params) or '(none)'}\n"
                f"    columns: {', '.join(v.columns)}"
            )
        return "\n".join(lines)
