"""fetch: the only graph node that calls get_data, via DataGateway (SPEC.md §6.1)."""

from __future__ import annotations

from ...data.canonical import CanonicalRequest
from ...memory.learning import CALL_ERROR, LearningEvent
from . import store_result


def run(ctx, state, config) -> dict:
    calls = state.get("calls_made", 0)
    result = ctx.gateway.fetch(
        state["view"], dict(state["params"]), user_id=state["user_id"],
        thread_id=state.get("thread_id", ""), calls_already_made=calls,
    )
    update: dict = {"calls_made": calls + 1}
    if result.ok:
        update.update(store_result.save(ctx, state, result))
        return update

    error = result.error
    note_ids = sorted({i for p in state.get("proposal", {}).values()
                       for i in p.get("note_ids", [])})
    ctx.learning.emit(LearningEvent.make(
        CALL_ERROR, view=state["view"], params=dict(state["params"]),
        error_class=type(error).__name__, message=str(error), note_ids=note_ids,
    ))
    key = CanonicalRequest.build(state["view"], state["params"]).key()
    update.update(
        last_error={"class": type(error).__name__, "message": str(error)},
        failed_keys=[*state.get("failed_keys", []), key],
        retry_approved=False,
    )
    return update


def route(state) -> str:
    return "diagnose" if state.get("last_error") else "respond"
