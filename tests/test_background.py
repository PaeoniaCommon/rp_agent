"""Learning runs in the background and never delays the reply (SPEC.md §8.6)."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from rp_agent.learning import LearningEvent

from .conftest import cfg
from .fakes import BlockingLLM, FakeLLM, params, view


def _llm():
    # node 'equities' -> a normalisation event, which the worker phrases with the LLM.
    return FakeLLM(ViewChoice=view("ViewLimits"), ParamExtraction=params(node="equities"))


def test_reply_does_not_wait_for_a_blocked_learning_model(make_agent):
    learn = BlockingLLM(NoteDraft={"text": "learned"})
    agent = make_agent(_llm(), learn_llm=learn)
    agent.chat("equities limits", cfg())
    assert learn.entered.wait(5), "worker never reached the learning model"

    started = time.perf_counter()
    reply = agent.chat("equities limits", cfg(thread="t2"), refresh=True)
    elapsed = time.perf_counter() - started
    assert reply.outcome == "new"
    assert elapsed < 2.0
    assert not learn.release.is_set()  # learning is still blocked
    learn.release.set()
    assert agent.flush(10)


def test_worker_exception_does_not_reach_the_user_or_stop_the_worker(make_agent):
    agent = make_agent(_llm())
    worker = agent.learning._worker
    real_handle = worker.handle
    calls = {"n": 0}

    def flaky(event):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        real_handle(event)

    worker.handle = flaky
    reply = agent.chat("equities limits", cfg())
    assert reply.outcome == "new"
    agent.flush(10)
    assert agent.learning.failed == 1
    assert agent.learning.processed >= 1


def test_full_queue_drops_events_without_blocking(make_agent):
    learn = BlockingLLM(NoteDraft={"text": "learned"})
    agent = make_agent(_llm(), learn_llm=learn, learning_queue_size=1)
    agent.chat("equities limits", cfg())
    learn.entered.wait(5)
    for i in range(5):
        reply = agent.chat("equities limits", cfg(thread=f"x{i}"), refresh=True)
        assert reply.outcome == "new"
    assert agent.learning.dropped > 0
    learn.release.set()
    agent.flush(10)


def test_flush_puts_notes_on_disk(make_agent, tmp_path):
    agent = make_agent(_llm())
    agent.chat("equities limits", cfg())
    assert agent.flush(10)
    assert "normalisation" in (tmp_path / "knowledge" / "params" / "node.md").read_text()


def test_close_drains_and_stops(make_agent):
    agent = make_agent(_llm())
    agent.chat("equities limits", cfg())
    agent.close()
    processed = agent.learning.processed
    assert processed >= 1
    agent.learning.emit(LearningEvent.make("call_ok", view="V", params={}))
    time.sleep(0.05)
    assert agent.learning.processed == processed


def test_events_never_carry_a_dataframe_or_user_id(make_agent):
    with pytest.raises(ValueError):
        LearningEvent.make("call_ok", user_id="U1")
    agent = make_agent(FakeLLM(ViewChoice=view("ViewUtilisations"),
                               ParamExtraction=params(node="equities",
                                                      current_date="2026-07-01")))
    seen = []
    real_emit = agent.learning.emit
    agent.learning.emit = lambda e: (seen.append(e), real_emit(e))
    agent.ctx.learning = agent.learning
    agent.chat("equities 1 July", cfg())

    def walk(value):
        assert not isinstance(value, pd.DataFrame)
        if isinstance(value, dict) or hasattr(value, "items"):
            for k, v in value.items():
                assert k not in ("user_id", "df")
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        else:
            assert value != "U000001"

    assert seen
    for event in seen:
        walk(event.payload)
