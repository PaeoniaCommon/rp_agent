"""check_cache: reuse the same user's identical request from the last 30 minutes (SPEC.md §7.3)."""

from __future__ import annotations

from ..canonical import CanonicalRequest


def run(ctx, state, config) -> dict:
    key = CanonicalRequest.build(state["view"], state["params"]).key(ctx.knowledge.defaults())
    if state.get("refresh"):
        return {"cache_key": key}
    now = ctx.clock()
    record = ctx.store.find_recent(state["user_id"], key, ctx.settings.reuse_window, now)
    if record is None:
        return {"cache_key": key}
    dataset = record.metadata()
    dataset["age_minutes"] = int((now - record.retrieved_at).total_seconds() // 60)
    return {"cache_key": key, "outcome": "reused", "dataset": dataset}


def route(state) -> str:
    return "respond" if state.get("outcome") == "reused" else "check_budget"
