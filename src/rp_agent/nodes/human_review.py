"""human_review: show the proposed call and wait for approve / edit / cancel (SPEC.md §5.6).

LangGraph re-runs a node from the top when it resumes after `interrupt`, so
everything before `interrupt()` here is pure; learning is emitted only after.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from .. import prompts
from ..learning import REVIEW, LearningEvent
from ..llm import structured
from ..schemas import ReviewDecision

_APPROVE = {"ok", "okay", "yes", "y", "approve", "approved", "go", "go ahead", "looks good",
            "correct", "fine", "proceed", "confirm"}
_CANCEL = {"cancel", "no", "n", "stop", "abort", "never mind", "nevermind"}
_PAIR_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:=|:|\s)\s*(.+?)\s*$")


# ----------------------------------------------------------------- payload

def build_payload(ctx, state) -> dict[str, Any]:
    view = state.get("view")
    status = state.get("param_status", {}) if view else {}
    issues = state.get("issues", [])
    last_error = state.get("last_error")
    view_certain = bool(view) and state.get("view_certain")
    view_options = [] if view_certain else [
        {"view": v, "description": ctx.catalogue.views[v].description}
        for v in ([view] if view else []) + state.get("view_alternatives", [])
        if v in ctx.catalogue.views
    ]
    params = []
    for name, s in status.items():
        params.append({"name": name, "value": s.get("value"), "status": s["status"],
                       "message": s.get("message", ""), "options": s.get("options", [])})
    payload = {
        "kind": "review",
        "role": "reviewer",
        "view": view,
        "view_status": "ok" if view_certain else "uncertain",
        "view_options": view_options,
        "params": params,
        "issues": issues,
        "error": last_error,
        "questions": state.get("questions", []),
    }
    payload["text"] = render(payload)
    return payload


def _fmt(value: Any) -> str:
    return "?" if value is None else repr(value)


def render(payload: dict[str, Any]) -> str:
    lines = []
    if payload.get("error"):
        lines.append(f"The query failed: {payload['error']['message']}")
        lines.append("I have not retried. Here is a corrected proposal for you to check.")
    else:
        lines.append("I need you to check this before I query.")
    lines.append("")
    if payload["view"] and payload["view_status"] == "ok":
        lines.append(f"  View:    {payload['view']}   ✓")
    else:
        lines.append(f"  View:    {payload['view'] or '?'}   ⚠ which data do you want?")
        for opt in payload["view_options"]:
            lines.append(f"             - {opt['view']} — {opt['description']}")
    for i, p in enumerate(payload["params"]):
        label = "  Params:  " if i == 0 else "           "
        mark = {"ok": "✓", "uncertain": "⚠", "missing": "?"}[p["status"]]
        line = f"{label}{p['name']} = {_fmt(p['value'])}   {mark}"
        if p["status"] == "missing":
            line += " (needed)"
        if p["message"] and p["status"] != "ok":
            line += f" {p['message']}"
        lines.append(line)
        if p["status"] != "ok" and p["options"]:
            lines.append(f"             options: {', '.join(map(str, p['options']))}"
                         + (" (or type another value)" if len(p["options"]) >= 10 else ""))
    for issue in payload["issues"]:
        lines.append(f"  ⚠ {issue['message']}")
        if issue.get("suggestions"):
            lines.append(f"    e.g. add one of: {', '.join(issue['suggestions'])}")
    for q in payload.get("questions", []):
        lines.append(f"  ? {q}")
    lines.append("")
    lines.append('Reply "ok" to approve, give corrections (e.g. "node EQUITIES"), or "cancel".')
    return "\n".join(lines)


# ------------------------------------------------------------------ parsing

def _coerce(ctx, name: str, raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip().strip("'\"")
    dtype = ctx.catalogue.params[name].dtype if name in ctx.catalogue.params else "str"
    if "int" in dtype and re.fullmatch(r"-?\d+", text):
        return int(text)
    if "float" in dtype and re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text) if "." in text else int(text)
    return text


def parse_reply(ctx, state, payload, reply: Any) -> dict[str, Any]:
    """{"action": ..., "edits": {name: value}} from a dict, plain text, or (fallback) the LLM."""
    view = state.get("view")
    view_params = set(ctx.catalogue.views[view].all_params) if view else set()

    if isinstance(reply, dict):
        edits = dict(reply.get("edits", {}))
        return {"action": reply.get("action", "edit" if edits else "approve"), "edits": edits}

    text = str(reply).strip()
    lowered = text.lower().rstrip(".!")
    if lowered in _APPROVE:
        return {"action": "approve", "edits": {}}
    if lowered in _CANCEL:
        return {"action": "cancel", "edits": {}}

    # A bare view name, or a bare value that is an option for exactly one open item.
    if text in ctx.catalogue.views:
        return {"action": "edit", "edits": {"view": text}}
    open_items = [p for p in payload["params"] if p["status"] != "ok"]
    matches = [p["name"] for p in open_items if text in map(str, p["options"])]
    if len(matches) == 1:
        return {"action": "edit", "edits": {matches[0]: _coerce(ctx, matches[0], text)}}

    # "name value" / "name=value" pairs, separated by commas, semicolons or new lines.
    edits: dict[str, Any] = {}
    for part in re.split(r"[,;\n]", text):
        if not part.strip():
            continue
        m = _PAIR_RE.match(part)
        if not m or (m.group(1) != "view" and m.group(1) not in view_params):
            edits = {}
            break
        edits[m.group(1)] = _coerce(ctx, m.group(1), m.group(2))
    if edits:
        return {"action": "edit", "edits": edits}

    decision = structured(ctx.llm, ReviewDecision, prompts.REVIEW_SYSTEM,
                          prompts.review_user(payload["text"], text))
    return {"action": decision.action,
            "edits": {e.name: _coerce(ctx, e.name, e.value) for e in decision.edits}}


# --------------------------------------------------------------------- node

def run(ctx, state, config) -> dict:
    payload = build_payload(ctx, state)
    reply = interrupt(payload)  # resumes with the reviewer's answer
    decision = parse_reply(ctx, state, payload, reply)

    update: dict[str, Any] = {"reviewed": True, "review": None}
    if decision["action"] == "cancel":
        update.update(outcome="cancelled")
        return update

    approved = set(state.get("approved", []))
    proposal = {k: dict(v) for k, v in state.get("proposal", {}).items()}
    status = state.get("param_status", {})
    view_before = state.get("view")
    view_after = view_before
    edits = decision["edits"]

    if "view" in edits and edits["view"] in ctx.catalogue.views:
        view_after = edits.pop("view")
    elif "view" in edits:
        edits.pop("view")
    view_changed = view_after != view_before

    edited = []
    if view_changed:
        approved = {"view"}
        keep = set(ctx.catalogue.views[view_after].all_params)
        proposal = {k: v for k, v in proposal.items() if k in keep}
    else:
        if view_after:
            approved.add("view")
        # "ok", or corrections to some items: everything else shown is approved as is.
        for name, s in status.items():
            if s.get("value") is not None:
                approved.add(name)
    for name, value in edits.items():
        if view_after and name in ctx.catalogue.views[view_after].all_params:
            proposal[name] = {"value": value, "source": "reviewer", "confidence": "high",
                              "note_ids": []}
            approved.add(name)
            edited.append(name)

    approved_uncertain = [n for n, s in status.items()
                          if s["status"] != "ok" and n not in edited and s.get("value") is not None]
    ctx.learning.emit(LearningEvent.make(
        REVIEW,
        request=state["request"],
        view_before=view_before,
        view_after=view_after,
        view_changed=view_changed and view_before is not None,
        view_was_uncertain=not state.get("view_certain"),
        view_note_ids=state.get("view_note_ids", []),
        edited=edited,
        approved_uncertain=approved_uncertain,
        params_after={k: v.get("value") for k, v in proposal.items()},
        note_ids_by_param={k: v.get("note_ids", []) for k, v in
                           state.get("proposal", {}).items()},
    ))

    # "ok" overrides warnings from learned rules; after an edit they are checked again.
    rules_overridden = not edited and (
        state.get("rules_overridden", False)
        or any(i.get("overridable") for i in state.get("issues", []))
    )
    update.update(
        view=view_after,
        view_certain=bool(view_after),
        proposal=proposal,
        approved=sorted(approved),
        rules_overridden=rules_overridden,
        # Approving after a failed call is what allows the one retry (§6.2).
        retry_approved=state.get("retry_approved", False) or bool(state.get("last_error")),
        last_error=None,
        needs_extract=view_changed or not proposal,
    )
    return update


def route(state) -> str:
    if state.get("outcome") == "cancelled":
        return "respond"
    if not state.get("view"):
        return "human_review"  # still no view: ask again
    return "extract_params" if state.get("needs_extract") else "validate_params"
