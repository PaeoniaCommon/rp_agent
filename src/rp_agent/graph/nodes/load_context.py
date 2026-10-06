"""load_context: confirm the request is well formed before any LLM work (SPEC.md §5.2).

The catalogue is built once at start-up; notes are read per node from the
in-memory knowledge base, so only the relevant ones reach each prompt.
"""

from __future__ import annotations


def run(ctx, state, config) -> dict:
    if not state.get("user_id"):
        # RPAgent.chat rejects this earlier; this is a second line of defence.
        return {"outcome": "error", "reply": "A user_id is required in the chat config."}
    return {}
