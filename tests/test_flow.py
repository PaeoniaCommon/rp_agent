"""End-to-end flows: one request one call, over-query protection, human review (SPEC.md §11)."""

from __future__ import annotations

import pytest

from .conftest import cfg
from .fakes import FakeLLM, params, view


def test_out_of_scope_makes_no_call(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view(None, single=False,
                                               parts=["limits", "utilisations"])))
    reply = agent.chat("limits and utilisations for EQUITIES", cfg())
    assert reply.outcome == "out_of_scope"
    assert "one data request at a time" in reply.text
    assert spy.calls == []


def test_certain_request_makes_one_call_and_no_review(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="EQUITIES", limit_level="Tier1")))
    reply = agent.chat("Tier1 limits for EQUITIES", cfg())
    assert reply.outcome == "new"
    assert not reply.waiting_for_review
    assert spy.calls == [("ViewLimits", {"node": "EQUITIES", "limit_level": "Tier1"})]
    ds = reply.dataset
    assert ds["view"] == "ViewLimits"
    assert ds["params"] == {"limit_level": "Tier1", "node": "EQUITIES"}
    assert ds["retrieved_at"].endswith("Z")
    assert ds["dataset_id"] in reply.text and "Retrieved at" in reply.text
    assert agent.store.get_df("U000001", ds["dataset_id"]).shape[0] == ds["row_count"]


def test_failed_call_is_only_retried_after_human_approval(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="MASTER")))
    reply = agent.chat("all limits", cfg())
    assert len(spy.calls) == 1  # timed out
    assert reply.waiting_for_review
    assert "timed out" in reply.text

    # Approving the same failing call does not repeat it.
    reply = agent.chat("ok", cfg())
    assert reply.waiting_for_review
    assert len(spy.calls) == 1

    reply = agent.chat("max_depth 1", cfg())
    assert reply.outcome == "new"
    assert spy.calls[-1] == ("ViewLimits", {"node": "MASTER", "max_depth": 1})
    assert len(spy.calls) == 2


def test_never_more_than_two_calls(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="MASTER")))
    agent.chat("all limits", cfg())
    reply = agent.chat("max_depth max", cfg())  # still too broad: second failure
    assert reply.outcome == "budget"
    assert "ViewTimeoutError" in reply.text
    assert len(spy.calls) == 2


def test_unknown_view_never_reaches_get_data(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view(None, confidence="low")))
    reply = agent.chat("something vague", cfg())
    assert reply.waiting_for_review
    assert "which data do you want" in reply.text
    assert spy.calls == []


def test_approving_with_no_view_asks_again(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view(None, confidence="low"),
                               ParamExtraction=params(node="FX")))
    agent.chat("something vague", cfg())
    reply = agent.chat("ok", cfg())
    assert reply.waiting_for_review and spy.calls == []
    reply = agent.chat("ViewLimits", cfg())
    assert reply.outcome == "new"
    assert spy.calls == [("ViewLimits", {"node": "FX"})]


def test_missing_mandatory_causes_review_and_no_call(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewUtilisations"),
                               ParamExtraction=params(node="EQUITIES", current_date=None)))
    reply = agent.chat("EQUITIES utilisation", cfg())
    assert reply.waiting_for_review
    statuses = {p["name"]: p["status"] for p in reply.review["params"]}
    assert statuses["current_date"] == "missing"
    assert spy.calls == []
    reply = agent.chat("current_date 2026-07-01", cfg())
    assert reply.outcome == "new"
    assert spy.calls == [("ViewUtilisations", {"node": "EQUITIES", "current_date": "2026-07-01"})]


def test_low_confidence_parameter_causes_review(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node=("EQUITIES", "guess", "low"))))
    reply = agent.chat("limits", cfg())
    assert reply.waiting_for_review and spy.calls == []


def test_closed_list_with_no_unique_fix_offers_full_list(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="EQUITIES", limit_level="Tier 3")))
    reply = agent.chat("Tier 3 limits for EQUITIES", cfg())
    assert reply.waiting_for_review and spy.calls == []
    level = next(p for p in reply.review["params"] if p["name"] == "limit_level")
    assert level["options"] == ["Tier0", "Tier1", "Warning"]


def test_case_is_normalised_before_the_call(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="equities")))
    reply = agent.chat("equities limits", cfg())
    assert reply.outcome == "new"
    assert spy.calls == [("ViewLimits", {"node": "EQUITIES"})]
    assert "'equities' → 'EQUITIES'" in reply.text


def test_unknown_node_goes_to_review_before_any_call(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="EQUITY")))
    reply = agent.chat("EQUITY limits", cfg())
    assert reply.waiting_for_review and spy.calls == []


def test_unambiguous_date_is_reformatted(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewUtilisations"),
                               ParamExtraction=params(node="FX", current_date="2026/07/01")))
    reply = agent.chat("FX utilisation 2026/07/01", cfg())
    assert reply.outcome == "new"
    assert spy.calls[0][1]["current_date"] == "2026-07-01"


def test_ambiguous_date_goes_to_review(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewUtilisations"),
                               ParamExtraction=params(node="FX", current_date="01/07/2026")))
    reply = agent.chat("FX utilisation 01/07/2026", cfg())
    assert reply.waiting_for_review and spy.calls == []
    date_item = next(p for p in reply.review["params"] if p["name"] == "current_date")
    assert date_item["options"] == ["2026-01-07", "2026-07-01"]


def test_reject_makes_no_call(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits", confidence="medium",
                                               alternatives=["ViewUtilisations"]),
                               ParamExtraction=params(node="EQUITIES")))
    reply = agent.chat("EQUITIES", cfg())
    assert reply.waiting_for_review
    reply = agent.chat("cancel", cfg())
    assert reply.outcome == "cancelled" and spy.calls == []


def test_review_edit_applies_values_and_view(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewLimits", confidence="medium",
                                  alternatives=["ViewUtilisations"]),
                  ParamExtraction=[params(node="EQUITIES"),
                                   params(node="EQUITIES", current_date="2026-07-01")])
    agent = make_agent(llm)
    reply = agent.chat("EQUITIES on 2026-07-01", cfg())
    assert reply.waiting_for_review
    reply = agent.chat("ViewUtilisations", cfg())
    assert reply.outcome == "new"
    assert spy.calls == [("ViewUtilisations", {"node": "EQUITIES",
                                                "current_date": "2026-07-01"})]


def test_free_text_review_uses_llm(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewLimits"),
                  ParamExtraction=params(node=("EQUITIES", "user_implied", "medium")),
                  ReviewDecision={"action": "edit",
                                  "edits": [{"name": "node", "value": "CREDIT"}]})
    agent = make_agent(llm)
    agent.chat("equity-ish limits", cfg())
    reply = agent.chat("actually I meant the credit desk", cfg())
    assert reply.outcome == "new"
    assert spy.calls == [("ViewLimits", {"node": "CREDIT"})]


def test_user_id_is_required(make_agent):
    agent = make_agent(FakeLLM())
    with pytest.raises(ValueError):
        agent.chat("x", {"configurable": {"thread_id": "t"}})
