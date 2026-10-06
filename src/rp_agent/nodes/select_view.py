"""select_view: deterministic checks on the proposed view (SPEC.md §5.3)."""

from __future__ import annotations


def run(ctx, state, config) -> dict:
    if not state.get("single_request", True):
        return {"outcome": "out_of_scope"}
    names = set(ctx.catalogue.view_names)
    view = state.get("view")
    if view not in names:
        view = None
    alternatives = [v for v in state.get("view_alternatives", []) if v in names]
    approved = "view" in state.get("approved", [])
    certain = approved or (
        view is not None and state.get("view_confidence") == "high" and not alternatives
    )
    if view is None:
        # Nothing to extract parameters for: a human must choose (§5.6).
        alternatives = alternatives or sorted(names)
    return {"view": view, "view_alternatives": alternatives, "view_certain": certain}


def route(state) -> str:
    if state.get("outcome") == "out_of_scope":
        return "respond"
    return "extract_params" if state.get("view") else "human_review"
