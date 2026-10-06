"""store_result: save the DataFrame under the user and get a dataset ID (SPEC.md §7).

Called by `fetch` in the same step, so the DataFrame never enters graph state
(it would otherwise be checkpointed).
"""

from __future__ import annotations

from ...data.canonical import CanonicalRequest
from ...data.gateway import FetchResult
from ...memory.learning import CALL_OK, LearningEvent, iso_dates_of


def save(ctx, state, result: FetchResult) -> dict:
    df = result.df
    request = CanonicalRequest.build(state["view"], state["params"])
    record = ctx.store.put(
        state["user_id"], df, request, result.retrieved_at, state["request"],
        thread_id=state.get("thread_id", ""), cache_key=state.get("cache_key", ""),
    )
    note_ids = sorted({i for p in state.get("proposal", {}).values()
                       for i in p.get("note_ids", [])} | set(state.get("view_note_ids", [])))
    ctx.learning.emit(LearningEvent.make(
        CALL_OK, view=state["view"], params=dict(state["params"]), request=state["request"],
        reviewed=state.get("reviewed", False), note_ids=note_ids, empty=df.empty,
        row_count=len(df), date_range=iso_dates_of(df),
    ))
    return {"outcome": "empty" if df.empty else "new", "dataset": record.metadata()}
