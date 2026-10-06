from __future__ import annotations

import pytest

from rp_agent import RPAgent, Settings

from .fakes import FakeClock, FakeLLM, GetDataSpy


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spy():
    return GetDataSpy()


@pytest.fixture
def make_agent(tmp_path, clock, spy):
    agents = []

    def _make(llm: FakeLLM, learn_llm=None, **settings):
        s = Settings(knowledge_dir=tmp_path / "knowledge", log_dir=tmp_path / "logs", **settings)
        agent = RPAgent(s, llm=llm, learn_llm=learn_llm or FakeLLM(), get_data=spy, clock=clock)
        agents.append(agent)
        return agent

    yield _make
    for agent in agents:
        agent.close()


def cfg(user="U000001", thread="t1"):
    return {"configurable": {"user_id": user, "thread_id": thread}}
