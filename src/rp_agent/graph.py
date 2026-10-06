"""Build and compile the LangGraph state graph (SPEC.md §5.1).

`get_data` is not an LLM tool: the only path to it is the `fetch` node, behind
validation, the cache and the budget (SPEC.md §3).
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .context import AgentContext
from .nodes import (
    check_budget,
    check_cache,
    check_scope,
    diagnose,
    extract_params,
    fetch,
    human_review,
    load_context,
    respond,
    select_view,
    validate_params,
)
from .state import AgentState

_NODES = {
    "load_context": load_context,
    "check_scope": check_scope,
    "select_view": select_view,
    "extract_params": extract_params,
    "validate_params": validate_params,
    "human_review": human_review,
    "check_cache": check_cache,
    "check_budget": check_budget,
    "fetch": fetch,
    "diagnose": diagnose,
    "respond": respond,
}


def build_graph(ctx: AgentContext, checkpointer=None):
    graph = StateGraph(AgentState)
    for name, module in _NODES.items():
        graph.add_node(name, partial(module.run, ctx))

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context",
        lambda s: "respond" if s.get("outcome") == "error" else "check_scope",
        ["respond", "check_scope"],
    )
    graph.add_conditional_edges("check_scope", check_scope.route, ["respond", "select_view"])
    graph.add_conditional_edges("select_view", select_view.route,
                                ["respond", "extract_params", "human_review"])
    graph.add_edge("extract_params", "validate_params")
    graph.add_conditional_edges("validate_params", validate_params.route,
                                ["human_review", "check_cache"])
    graph.add_conditional_edges("human_review", human_review.route,
                                ["respond", "human_review", "extract_params", "validate_params"])
    graph.add_conditional_edges("check_cache", check_cache.route, ["respond", "check_budget"])
    graph.add_conditional_edges("check_budget", check_budget.route, ["respond", "fetch"])
    graph.add_conditional_edges("fetch", fetch.route, ["respond", "diagnose"])
    graph.add_conditional_edges("diagnose", diagnose.route, ["respond", "human_review"])
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
