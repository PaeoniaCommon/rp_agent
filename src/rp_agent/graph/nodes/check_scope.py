"""check_scope: one LLM call that checks scope and proposes a view (SPEC.md §5.2, §5.3).

Scope and view choice share one structured call to keep latency down;
`select_view` then applies the deterministic checks to the proposal.
"""

from __future__ import annotations

from ...llm import prompts
from ...llm.schemas import view_choice_model


def run(ctx, state, config) -> dict:
    kb = ctx.knowledge
    names = tuple(ctx.catalogue.view_names)
    general = kb.rules("general", "general")
    view_rules = {v: kb.rules("view", v) for v in names}
    examples = {v: kb.examples(v) for v in names}
    user = prompts.view_user(state["request"], ctx.catalogue.describe_views(), general,
                             view_rules, examples, ctx.today())
    choice = ctx.llm.structured(view_choice_model(names), prompts.VIEW_SYSTEM, user)
    update = {} if choice.single_request else {"outcome": "out_of_scope"}
    return {
        **update,
        "single_request": bool(choice.single_request),
        "parts": list(choice.parts or []),
        "view": choice.view,
        "view_confidence": choice.confidence,
        "view_alternatives": [v for v in choice.alternatives if v != choice.view],
        "view_reason": choice.reason,
        "view_note_ids": list(choice.note_ids or []),
    }


def route(state) -> str:
    if state.get("outcome") == "error":
        return "respond"
    return "select_view" if state.get("single_request", True) else "respond"
