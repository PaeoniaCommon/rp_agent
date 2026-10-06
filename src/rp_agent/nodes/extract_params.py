"""extract_params: propose the view's parameters (SPEC.md §5.4).

Loads only the notes for the chosen view and its parameters, and shows each
parameter's values sized by its value domain (SPEC.md §8.4).
"""

from __future__ import annotations

from .. import prompts
from ..llm import structured
from ..schemas import param_extraction_model
from ..values import CLOSED_UNPUBLISHED, LARGE_CLOSED, SMALL_CLOSED


def _values_text(ctx, name: str, request: str) -> str:
    domain = ctx.domains.domain(name)
    if domain == SMALL_CLOSED:
        return "exactly one of " + ", ".join(ctx.domains.allowed_values(name))
    if domain in (LARGE_CLOSED, CLOSED_UNPUBLISHED):
        shortlist = ctx.domains.shortlist(name, request, ctx.settings.prompt_shortlist_size)
        if shortlist:
            return ("closed list (partial; closest known matches): " + ", ".join(shortlist))
        return "closed list (no known values yet)"
    return ""


def run(ctx, state, config) -> dict:
    kb = ctx.knowledge
    view = ctx.catalogue.views[state["view"]]
    request = state["request"]
    blocks = []
    for name in view.all_params:
        blocks.append(prompts.param_block(
            name, name in view.mandatory_params, ctx.catalogue.params[name],
            ctx.domains.domain(name), _values_text(ctx, name, request),
            kb.aliases_in_text(name, request), kb.rules("param", name)))
    user = prompts.params_user(request, view, blocks, kb.rules("general", "general"),
                               kb.rules("view", view.name), kb.examples(view.name), ctx.today())
    schema = param_extraction_model(tuple(view.all_params))
    extraction = structured(ctx.llm, schema, prompts.PARAMS_SYSTEM, user)

    proposal: dict[str, dict] = {}
    for p in extraction.params:
        if p.value is None and p.name not in view.mandatory_params:
            continue
        proposal[p.name] = {"value": p.value, "source": p.source,
                            "confidence": p.confidence, "note_ids": list(p.note_ids or [])}

    # Values a human set or approved in review always win (§5.6).
    previous = state.get("proposal", {})
    for name in state.get("approved", []):
        if name in previous and name in view.all_params:
            proposal[name] = previous[name]
    return {"proposal": proposal, "questions": list(extraction.questions or [])}
