"""respond: the reply to the user (SPEC.md §9.3)."""

from __future__ import annotations

from typing import Any


def _params_text(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items()) or "(none)"


def _dataset_lines(dataset: dict[str, Any], with_shape: bool) -> list[str]:
    lines = [
        f"  View:          {dataset['view']}",
        f"  Params:        {_params_text(dataset['params'])}",
        f"  Retrieved at:  {dataset['retrieved_at']}",
    ]
    if with_shape:
        lines.append(f"  Result:        {dataset['row_count']} rows × "
                     f"{len(dataset['columns'])} columns")
    return lines


def run(ctx, state, config) -> dict:
    outcome = state.get("outcome")
    dataset = state.get("dataset")
    adjustments = state.get("adjustments", [])
    if outcome in ("new", "empty"):
        head = f"Retrieved dataset {dataset['dataset_id']} (new query)"
        lines = [head, *_dataset_lines(dataset, with_shape=True)]
        if outcome == "empty":
            lines.append("  Note:          the query was valid but returned no rows.")
        if adjustments:
            lines.append(f"  Adjustments:   {'; '.join(adjustments)}")
        reply = "\n".join(lines)
    elif outcome == "reused":
        minutes = dataset.get("age_minutes", 0)
        head = (f"Reusing dataset {dataset['dataset_id']} — you retrieved the same view and "
                f"parameters {minutes} minute{'s' if minutes != 1 else ''} ago (no new query)")
        reply = "\n".join([head, *_dataset_lines(dataset, with_shape=False)])
    elif outcome == "out_of_scope":
        parts = state.get("parts") or []
        listed = f" ({'; '.join(parts)})" if parts else ""
        reply = (f"This asks for more than one dataset{listed}. I handle one data request at "
                 "a time, so please send them as separate requests. No data was queried.")
    elif outcome == "cancelled":
        reply = "Cancelled. No data was queried."
    elif outcome == "budget":
        reply = ("I have used the get_data calls allowed for this request and have not "
                 "queried again. Please send a new request.")
        if error := state.get("last_error"):
            reply = f"The query failed ({error['class']}: {error['message']}). " + reply
    else:
        reply = state.get("reply") or "Something went wrong; no data was retrieved."
    return {"reply": reply, "review": None}
