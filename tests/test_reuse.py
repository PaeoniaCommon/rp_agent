"""Store and 30-minute reuse, per user (SPEC.md §7, §11)."""

from __future__ import annotations

import pytest

from .conftest import cfg
from .fakes import FakeLLM, params, view


def _agent(make_agent, **values):
    return make_agent(FakeLLM(ViewChoice=view("ViewUtilisations"),
                              ParamExtraction=params(**values)))


def test_same_user_other_thread_within_30_minutes_reuses(make_agent, spy, clock):
    agent = _agent(make_agent, node="EQUITIES", current_date="2026-07-01")
    first = agent.chat("EQ on 1 July", cfg(thread="a"))
    clock.advance(minutes=29)
    second = agent.chat("EQ on 1 July", cfg(thread="b"))
    assert second.outcome == "reused"
    assert second.dataset["dataset_id"] == first.dataset["dataset_id"]
    assert "29 minutes ago" in second.text
    assert len(spy.calls) == 1


def test_other_user_makes_a_new_call_and_cannot_read_first_users_data(make_agent, spy):
    agent = _agent(make_agent, node="EQUITIES", current_date="2026-07-01")
    first = agent.chat("EQ on 1 July", cfg(user="U1"))
    second = agent.chat("EQ on 1 July", cfg(user="U2"))
    assert second.outcome == "new"
    assert len(spy.calls) == 2
    with pytest.raises(KeyError):
        agent.store.get("U2", first.dataset["dataset_id"])


def test_after_31_minutes_a_new_call_is_made(make_agent, spy, clock):
    agent = _agent(make_agent, node="EQUITIES", current_date="2026-07-01")
    first = agent.chat("EQ on 1 July", cfg())
    clock.advance(minutes=31)
    second = agent.chat("EQ on 1 July", cfg())
    assert second.outcome == "new"
    assert second.dataset["dataset_id"] != first.dataset["dataset_id"]
    assert agent.store.get("U000001", first.dataset["dataset_id"])
    assert len(spy.calls) == 2


def test_int_and_float_forms_hit_the_cache(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewUtilisations"), ParamExtraction=[
        params(node="EQUITIES", current_date="2026-07-01", min_utilisation=85),
        params(min_utilisation=85.0, current_date="2026-07-01", node="EQUITIES"),
    ])
    agent = make_agent(llm)
    agent.chat("EQ above 85%", cfg())
    assert agent.chat("EQ above 85%", cfg()).outcome == "reused"
    assert len(spy.calls) == 1


def test_explicit_refresh_skips_the_cache(make_agent, spy):
    agent = _agent(make_agent, node="EQUITIES", current_date="2026-07-01")
    first = agent.chat("EQ on 1 July", cfg())
    second = agent.chat("EQ on 1 July", cfg(), refresh=True)
    assert second.outcome == "new" and len(spy.calls) == 2
    assert agent.store.get("U000001", first.dataset["dataset_id"])
    third = agent.chat("refresh EQ on 1 July", cfg())
    assert third.outcome == "new" and len(spy.calls) == 3


def test_empty_result_is_stored_with_an_id(make_agent, spy):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="RATES_EM")))
    reply = agent.chat("RATES_EM limits", cfg())
    assert reply.outcome == "empty"
    assert reply.dataset["row_count"] == 0
    assert "returned no rows" in reply.text
    df = agent.store.get_df("U000001", reply.dataset["dataset_id"])
    assert df.empty and list(df.columns) == reply.dataset["columns"]
