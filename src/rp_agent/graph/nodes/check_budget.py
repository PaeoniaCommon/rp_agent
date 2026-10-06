"""check_budget: at most one automatic call, and a hard cap of two (SPEC.md §6.2)."""

from __future__ import annotations


def run(ctx, state, config) -> dict:
    calls = state.get("calls_made", 0)
    if calls >= ctx.settings.max_calls_per_request:
        return {"outcome": "budget"}
    if calls >= ctx.settings.max_auto_calls_per_request and not state.get("retry_approved"):
        return {"outcome": "budget"}
    return {}


def route(state) -> str:
    return "respond" if state.get("outcome") == "budget" else "fetch"
