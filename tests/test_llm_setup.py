"""Model setup: langchain_openai.ChatOpenAI with model, API key and base URL (SPEC.md §2)."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from rp_agent import RPAgent, Settings
from rp_agent.llm import LLMClient, make_chat_model

from .conftest import cfg
from .fakes import FakeLLM, params, view


def test_make_chat_model_uses_model_key_and_base_url():
    llm = make_chat_model("my-model", api_key="sk-test", base_url="https://llm.example.com/v1")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "my-model"
    assert llm.openai_api_key.get_secret_value() == "sk-test"
    assert llm.openai_api_base == "https://llm.example.com/v1"


def test_make_chat_model_needs_a_model():
    with pytest.raises(ValueError, match="No model configured"):
        make_chat_model("", api_key="sk-test")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RP_AGENT_MODEL", "my-model")
    monkeypatch.setenv("RP_AGENT_API_KEY", "sk-secret")
    monkeypatch.setenv("RP_AGENT_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("RP_AGENT_STRUCTURED_OUTPUT", "function_calling")
    monkeypatch.setenv("RP_AGENT_TEMPERATURE", "0")
    s = Settings.from_env()
    assert (s.model, s.api_key, s.base_url) == ("my-model", "sk-secret",
                                                 "https://llm.example.com/v1")
    assert s.structured_output_method == "function_calling"
    assert s.temperature == 0.0
    assert "sk-secret" not in repr(s)
    assert s.learn_uses_main_model


def test_agent_builds_chat_openai_from_settings(tmp_path):
    s = Settings(model="my-model", api_key="sk-test", base_url="https://llm.example.com/v1",
                 knowledge_dir=tmp_path / "k", log_dir=tmp_path / "l")
    agent = RPAgent(s)
    try:
        assert isinstance(agent.ctx.llm.chat, ChatOpenAI)
        assert agent.ctx.llm.chat.openai_api_base == "https://llm.example.com/v1"
        assert agent.ctx.llm.method == "json_schema"
        assert agent.learning._worker.llm.chat is agent.ctx.llm.chat
    finally:
        agent.close()


def test_separate_learning_model(tmp_path):
    s = Settings(model="big", api_key="k1", base_url="https://a.example.com/v1",
                 learn_model="small", learn_base_url="https://b.example.com/v1",
                 knowledge_dir=tmp_path / "k", log_dir=tmp_path / "l")
    agent = RPAgent(s)
    try:
        learn = agent.learning._worker.llm.chat
        assert learn.model_name == "small"
        assert learn.openai_api_base == "https://b.example.com/v1"
        assert learn.openai_api_key.get_secret_value() == "k1"  # falls back to the main key
    finally:
        agent.close()


def test_structured_output_method_is_passed_through(make_agent):
    llm = FakeLLM(ViewChoice=view("ViewLimits"), ParamExtraction=params(node="FX"))
    agent = make_agent(llm, structured_output_method="function_calling")
    assert agent.chat("FX limits", cfg()).outcome == "new"
    assert set(llm.methods) == {"function_calling"}


def test_bad_structured_output_method_is_rejected():
    with pytest.raises(ValueError):
        LLMClient(FakeLLM(), method="xml")
