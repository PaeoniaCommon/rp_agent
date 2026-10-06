"""Learning from errors and reviews, and value domains (SPEC.md §8, §11)."""

from __future__ import annotations

import json

from rp_agent.memory.learning import CALL_ERROR, LearningEvent
from rp_agent.memory.values import CLOSED_UNPUBLISHED

from .conftest import cfg
from .fakes import FakeLLM, params, view


def test_knowledge_dir_is_created_empty(make_agent, tmp_path):
    make_agent(FakeLLM())
    root = tmp_path / "knowledge"
    assert root.is_dir() and list(root.iterdir()) == []


def test_master_timeout_is_learned_and_reviewed_before_any_call_next_time(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewLimits"), ParamExtraction=params(node="MASTER"))
    first = make_agent(llm)
    reply = first.chat("all limits", cfg())
    assert len(spy.calls) == 1 and reply.waiting_for_review
    first.flush()
    rules = first.knowledge.rules("view", "ViewLimits", "requires_filter")
    assert rules and rules[0]["certainty"] == "certain"

    second = make_agent(llm)  # a new session sharing the knowledge directory
    reply = second.chat("all limits", cfg(thread="new"))
    assert reply.waiting_for_review
    assert "time out" in reply.text
    assert len(spy.calls) == 1  # no new call


def test_alias_from_review_removes_the_review_next_time(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewLimits"), ParamExtraction=[
        params(node=None),
        params(node=("EQUITIES", "note", "medium")),
    ])
    learn = FakeLLM(PhraseDraft={"phrase": "equity desk"}, NoteDraft={"text": "n"})
    agent = make_agent(llm, learn_llm=learn)
    reply = agent.chat("limits for the equity desk", cfg())
    assert reply.waiting_for_review
    reply = agent.chat("node EQUITIES", cfg())
    assert reply.outcome == "new"
    agent.flush()
    alias = agent.knowledge.aliases("node")[0]
    assert alias["phrase"] == "equity desk" and alias["certainty"] == "likely"

    # Another user (so the 30-minute reuse does not apply): no review this time.
    reply = agent.chat("show me the equity desk limits", cfg(user="U2"))
    assert reply.outcome == "new"
    assert len(spy.calls) == 2


def test_failed_call_demotes_the_notes_it_followed(make_agent):
    agent = make_agent(FakeLLM())
    kb = agent.knowledge
    alias_id = kb.add_alias("node", "desk", "EQUITIES", "likely", "human_review")
    agent.learning.emit(LearningEvent.make(
        CALL_ERROR, view="ViewLimits", params={"node": "EQUITIES"},
        error_class="InvalidParameterError", message="ViewLimits: x", note_ids=[alias_id]))
    agent.flush()
    assert kb.find_note(alias_id)["certainty"] == "tentative"


def test_notes_stay_within_limits_and_hold_no_user_or_data(make_agent, tmp_path):
    agent = make_agent(FakeLLM())
    kb = agent.knowledge
    kb.add_rule("param", "limit_id", "format", "human-written", "tentative", "human")
    for i in range(50):
        agent.learning.emit(LearningEvent.make(
            CALL_ERROR, view="ViewLimits", params={"node": "FX", "limit_id": f"LIM{i:06d}"},
            error_class="InvalidParameterError",
            message=f"ViewLimits: limit_id='bad{i}' must match ^LIM\\d{{6}}$", note_ids=[]))
    agent.flush()
    rules = kb.rules("param", "limit_id")
    assert len(rules) == agent.settings.max_rules_per_file
    assert any(r["source"] == "human" for r in rules)
    for path in (tmp_path / "knowledge").rglob("*"):
        if path.is_file():
            assert "U000001" not in path.read_text()


def test_small_closed_lists_are_shown_in_full_and_never_stored(make_agent, tmp_path):
    llm = FakeLLM(ViewChoice=view("ViewLimits"),
                  ParamExtraction=params(node="EQUITIES", limit_level="Tier1"))
    agent = make_agent(llm)
    agent.chat("Tier1 limits for EQUITIES", cfg())
    agent.flush()
    prompt = next(u for n, u in llm.calls if n == "ParamExtraction")
    assert "exactly one of Tier0, Tier1, Warning" in prompt
    assert "EQ Delta" in prompt
    values_dir = tmp_path / "knowledge" / "values"
    assert not (values_dir / "limit_level.jsonl").exists()
    assert not (values_dir / "risk_factor.jsonl").exists()


def test_node_becomes_closed_and_valid_nodes_are_indexed(make_agent):
    llm = FakeLLM(ViewChoice=view("ViewLimits"),
                  ParamExtraction=[params(node="equities"), params(node="CREDIT")])
    agent = make_agent(llm)
    agent.chat("equities limits", cfg())
    agent.flush()
    assert agent.ctx.domains.domain("node") == CLOSED_UNPUBLISHED
    agent.chat("CREDIT limits", cfg())
    agent.flush()
    assert set(agent.ctx.domains.index("node").values()) == {"EQUITIES", "CREDIT"}


def test_long_lists_are_shortlisted_in_prompt_and_review(make_agent, tmp_path):
    llm = FakeLLM(ViewChoice=view("ViewLimits"),
                  ParamExtraction=params(node=("EQUITY", "user_implied", "medium")))
    agent = make_agent(llm)
    agent.knowledge.set_domain("node", CLOSED_UNPUBLISHED)
    index = agent.ctx.domains.index("node")
    for i in range(1000):
        index.add(f"DESK_{i:04d}", used=False)
    for v in ["EQUITIES", "EQ_EMEA", "EQ_US", "EQ_APAC", "EQ_EMEA_CASH"]:
        index.add(v)
    index.save()

    reply = agent.chat("limits for the equity book", cfg())
    prompt = next(u for n, u in llm.calls if n == "ParamExtraction")
    node_line = next(line for line in prompt.splitlines() if "closed list (partial" in line)
    assert len(node_line.split(":", 2)[-1].split(",")) <= 15
    assert "EQUITIES" in node_line
    item = next(p for p in reply.review["params"] if p["name"] == "node")
    assert 0 < len(item["options"]) <= 10
    node_md = tmp_path / "knowledge" / "params" / "node.md"
    assert node_md.stat().st_size < 2048


def test_value_not_in_index_is_accepted_if_rpdata_accepts_it(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewLimits"), ParamExtraction=params(node="FX_SPOT"))
    agent = make_agent(llm)
    agent.knowledge.set_domain("node", CLOSED_UNPUBLISHED)
    assert agent.chat("FX spot limits", cfg()).outcome == "new"
    assert spy.calls == [("ViewLimits", {"node": "FX_SPOT"})]


def test_open_parameters_store_no_values(make_agent, tmp_path):
    llm = FakeLLM(ViewChoice=view("ViewLimits"),
                  ParamExtraction=params(node="FX", limit_group="LG_FX"))
    agent = make_agent(llm)
    agent.chat("FX limits for LG_FX", cfg())
    agent.flush()
    values_dir = tmp_path / "knowledge" / "values"
    for name in ("limit_group", "limit_id", "user_id"):
        assert not (values_dir / f"{name}.jsonl").exists()


def test_weekend_empty_result_is_learned(make_agent, spy):
    llm = FakeLLM(ViewChoice=view("ViewUtilisations"),
                  ParamExtraction=params(node="FX", current_date="2026-07-04"))
    agent = make_agent(llm)
    assert agent.chat("FX on Saturday 4 July", cfg()).outcome == "empty"
    agent.flush()
    assert any(r["dedup_key"] == "coverage:weekend"
               for r in agent.knowledge.rules("general", "general", "coverage"))
    reply = agent.chat("FX on Saturday 4 July", cfg(thread="t2"), refresh=True)
    assert reply.waiting_for_review and "weekend" in reply.text
    assert len(spy.calls) == 1


def test_call_log_is_written(make_agent, tmp_path):
    agent = make_agent(FakeLLM(ViewChoice=view("ViewLimits"),
                               ParamExtraction=params(node="FX")))
    agent.chat("FX limits", cfg())
    lines = (tmp_path / "logs" / "get_data_calls.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["view"] == "ViewLimits" and entry["outcome"] == "ok"
    assert entry["user_id"] == "U000001"
