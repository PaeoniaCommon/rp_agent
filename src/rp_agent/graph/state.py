"""LangGraph state for one data request (SPEC.md §5)."""

from __future__ import annotations

from typing import Any, TypedDict


class ParamProposal(TypedDict, total=False):
    value: Any
    source: str  # user_explicit | user_implied | note | default | guess | reviewer
    confidence: str  # high | medium | low
    note_ids: list[str]


class AgentState(TypedDict, total=False):
    # Input
    request: str
    refresh: bool
    user_id: str
    thread_id: str

    # View (SPEC.md §5.3)
    single_request: bool
    parts: list[str]
    view: str | None
    view_confidence: str
    view_alternatives: list[str]
    view_reason: str
    view_note_ids: list[str]
    view_certain: bool

    # Parameters (SPEC.md §5.4, §5.5)
    proposal: dict[str, ParamProposal]
    questions: list[str]
    params: dict[str, Any]  # what would be sent to get_data
    param_status: dict[str, dict[str, Any]]
    issues: list[dict[str, Any]]
    adjustments: list[str]
    approved: list[str]  # items a human approved or set in review ("view", param names)
    rules_overridden: bool  # a human approved despite a learned-rule warning
    needs_review: bool
    needs_extract: bool
    reviewed: bool

    # Calling get_data (SPEC.md §6)
    calls_made: int
    retry_approved: bool
    failed_keys: list[str]
    last_error: dict[str, Any] | None
    cache_key: str

    # Outcome
    outcome: str  # new | reused | empty | out_of_scope | cancelled | budget | error
    dataset: dict[str, Any] | None
    review: dict[str, Any] | None
    reply: str


def initial_state(request: str, user_id: str, thread_id: str, refresh: bool) -> AgentState:
    """Every key is set, so a new request in an existing thread starts clean."""
    return AgentState(
        request=request, refresh=refresh, user_id=user_id, thread_id=thread_id,
        single_request=True, parts=[], view=None, view_confidence="low",
        view_alternatives=[], view_reason="", view_note_ids=[], view_certain=False,
        proposal={}, questions=[], params={}, param_status={}, issues=[], adjustments=[],
        approved=[], rules_overridden=False, needs_review=False, needs_extract=False, reviewed=False,
        calls_made=0, retry_approved=False, failed_keys=[], last_error=None, cache_key="",
        outcome="", dataset=None, review=None, reply="",
    )
