"""diagnose: classify a failed call and build a corrected proposal for review (SPEC.md §6.3).

The agent never retries by itself: a retry needs a human to approve the proposal.
"""

from __future__ import annotations

import re

_BUG_ERRORS = {"UnknownViewError", "UnexpectedParameterError", "MissingParameterError"}
_RETRYABLE = {"InvalidParameterError", "UnknownNodeError", "ViewTimeoutError"}


def run(ctx, state, config) -> dict:
    error = state["last_error"]
    cls, message = error["class"], error["message"]
    if cls in _BUG_ERRORS or cls not in _RETRYABLE:
        return {"outcome": "error",
                "reply": f"The query failed ({cls}: {message}). I have not retried."}
    if state.get("calls_made", 0) >= ctx.settings.max_calls_per_request:
        return {"outcome": "budget"}

    status = {k: dict(v) for k, v in state.get("param_status", {}).items()}
    issues = []
    view = ctx.catalogue.views[state["view"]]
    if cls == "ViewTimeoutError":
        suggestions = [p for p in view.optional_params if p not in state["params"]]
        if "max_depth" in view.optional_params:
            suggestions = ["max_depth (an integer)" if p == "max_depth" else p
                           for p in suggestions]
            if state["params"].get("max_depth") == "max":
                suggestions.insert(0, "max_depth (an integer)")
        issues.append({"key": "timeout", "overridable": False, "params": [],
                       "message": "The query was too broad and timed out. Add a filter.",
                       "suggestions": list(dict.fromkeys(suggestions))})
    else:
        named = re.search(r":\s*(\w+)=", message)
        name = named.group(1) if named else None
        if name in status:
            status[name].update(
                status="uncertain", message=message.split(": ", 1)[-1],
                options=ctx.validator.options_for(name, state["request"],
                                                  status[name].get("value")))
        else:
            issues.append({"key": "error", "overridable": False, "params": [],
                           "message": message, "suggestions": []})
    return {"param_status": status, "issues": issues, "needs_review": True}


def route(state) -> str:
    return "respond" if state.get("outcome") in ("error", "budget") else "human_review"
