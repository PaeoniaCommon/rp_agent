"""validate_params: pre-flight checks, no get_data call (SPEC.md §5.5)."""

from __future__ import annotations


def run(ctx, state, config) -> dict:
    result = ctx.validator.check(
        view=state["view"],
        proposal=state.get("proposal", {}),
        request=state["request"],
        approved=set(state.get("approved", [])),
        rules_overridden=state.get("rules_overridden", False),
        failed_keys=state.get("failed_keys", []),
    )
    for event in result.events:
        ctx.learning.emit(event)  # non-blocking (§8.6)

    # Keep note IDs alongside the proposal for confirming / demoting notes later.
    proposal = {name: dict(p) for name, p in state.get("proposal", {}).items()}
    for name, ids in result.note_ids.items():
        if name in proposal:
            proposal[name]["note_ids"] = ids

    needs_review = not state.get("view_certain") or not result.all_certain
    return {
        "params": result.params,
        "param_status": result.status,
        "issues": result.issues,
        "adjustments": _merge(state.get("adjustments", []), result.adjustments),
        "proposal": proposal,
        "needs_review": needs_review,
    }


def _merge(old: list[str], new: list[str]) -> list[str]:
    return list(dict.fromkeys([*old, *new]))


def route(state) -> str:
    return "human_review" if state.get("needs_review") else "check_cache"
