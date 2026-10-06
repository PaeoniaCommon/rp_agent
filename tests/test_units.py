"""Unit tests for the store, canonical keys, value index, shortlist and knowledge base."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from rp_agent.config import Settings
from rp_agent.data.canonical import CanonicalRequest
from rp_agent.data.catalogue import Catalogue
from rp_agent.data.store import InMemoryDataStore
from rp_agent.memory.knowledge import KnowledgeBase
from rp_agent.memory.values import (
    CLOSED_UNPUBLISHED,
    OPEN,
    SMALL_CLOSED,
    ValueDomains,
    ValueIndex,
)

from .fakes import FakeClock


def test_canonical_key_ignores_order_none_and_int_float():
    a = CanonicalRequest.build("V", {"b": 85, "a": "x", "c": None})
    b = CanonicalRequest.build("V", {"a": "x", "b": 85.0})
    assert a.key() == b.key()
    assert a.params_dict == {"a": "x", "b": 85}


def test_canonical_key_uses_certain_defaults():
    with_default = CanonicalRequest.build("V", {"node": "X", "max_depth": "max"})
    without = CanonicalRequest.build("V", {"node": "X"})
    assert with_default.key() != without.key()
    assert with_default.key({"max_depth": "max"}) == without.key({"max_depth": "max"})


def test_store_is_per_user_and_returns_copies():
    clock = FakeClock()
    store = InMemoryDataStore(clock)
    req = CanonicalRequest.build("ViewLimits", {"node": "EQUITIES"})
    rec = store.put("U1", pd.DataFrame({"a": [1]}), req, clock(), "req", cache_key=req.key())
    assert rec.dataset_id.startswith("ds_") and len(rec.dataset_id) == 11
    with pytest.raises(KeyError):
        store.get("U2", rec.dataset_id)
    df = store.get_df("U1", rec.dataset_id)
    df.loc[0, "a"] = 99
    assert store.get_df("U1", rec.dataset_id).loc[0, "a"] == 1
    assert store.find_recent("U1", req.key(), timedelta(minutes=30), clock()) is not None
    assert store.find_recent("U2", req.key(), timedelta(minutes=30), clock()) is None
    clock.advance(minutes=31)
    assert store.find_recent("U1", req.key(), timedelta(minutes=30), clock()) is None
    assert store.get("U1", rec.dataset_id).dataset_id == rec.dataset_id


def test_value_index_cap_evicts_least_recently_used(tmp_path):
    clock = FakeClock()
    index = ValueIndex(tmp_path / "v.jsonl", cap=3, clock=clock)
    for i, v in enumerate(["A", "B", "C"]):
        clock.advance(days=1)
        index.add(v)
    clock.advance(days=1)
    index.add("A")  # A is now most recent
    index.add("D")
    assert sorted(index.values()) == ["A", "C", "D"]
    index.save()
    assert sorted(ValueIndex(tmp_path / "v.jsonl", cap=3, clock=clock).values()) == [
        "A", "C", "D"]


@pytest.fixture
def domains(tmp_path):
    clock = FakeClock()
    kb = KnowledgeBase(tmp_path / "k", clock)
    return ValueDomains(Catalogue(), kb, Settings(knowledge_dir=tmp_path / "k"), clock), kb


def test_domains_classification(domains):
    d, kb = domains
    assert d.domain("limit_level") == SMALL_CLOSED
    assert d.domain("risk_factor") == SMALL_CLOSED
    assert d.domain("node") == OPEN  # until rpdata rejects an unknown well-formed node
    assert d.domain("limit_id") == OPEN
    kb.set_domain("node", CLOSED_UNPUBLISHED)
    assert d.domain("node") == CLOSED_UNPUBLISHED


def test_shortlist_is_bounded_and_relevant(domains):
    d, kb = domains
    kb.set_domain("node", CLOSED_UNPUBLISHED)
    index = d.index("node")
    for i in range(1000):
        index.add(f"DESK_{i:04d}", used=False)
    for v in ["EQUITIES", "EQ_EMEA", "EQ_US", "CREDIT"]:
        index.add(v)
    shortlist = d.shortlist("node", "limits for the equity book", 15)
    assert len(shortlist) <= 15
    assert "EQUITIES" in shortlist
    assert "EQ_EMEA" in shortlist


def test_knowledge_rule_dedup_limits_and_human_notes(tmp_path):
    clock = FakeClock()
    kb = KnowledgeBase(tmp_path / "k", clock, max_rules_per_file=5)
    human = kb.add_rule("param", "limit_id", "format", "human note", "tentative", "human")
    first = kb.add_rule("param", "limit_id", "format", "a", "tentative", "rpdata_error",
                        dedup_key="k1")
    again = kb.add_rule("param", "limit_id", "format", "a2", "certain", "rpdata_error",
                        dedup_key="k1")
    assert first == again
    assert kb.find_note(first)["certainty"] == "certain"
    for i in range(20):
        kb.add_rule("param", "limit_id", "format", f"r{i}", "tentative", "rpdata_error")
    rules = kb.rules("param", "limit_id")
    assert len(rules) == 5
    assert any(r["id"] == human for r in rules)
    kb.demote_notes([human])
    assert kb.find_note(human) is not None  # human notes are never demoted away


def test_knowledge_promote_and_demote(tmp_path):
    kb = KnowledgeBase(tmp_path / "k", FakeClock())
    rid = kb.add_rule("view", "ViewLimits", "hint", "x", "tentative", "success")
    kb.confirm([rid])
    assert kb.find_note(rid)["certainty"] == "tentative"
    kb.confirm([rid])
    assert kb.find_note(rid)["certainty"] == "likely"
    kb.demote_notes([rid])
    assert kb.find_note(rid)["certainty"] == "tentative"
    kb.demote_notes([rid])
    assert kb.find_note(rid) is None


def test_knowledge_reloads_from_disk(tmp_path):
    clock = FakeClock()
    kb = KnowledgeBase(tmp_path / "k", clock)
    kb.add_alias("node", "equity desk", "EQUITIES", "likely", "human_review")
    kb2 = KnowledgeBase(tmp_path / "k", clock)
    assert kb2.aliases_in_text("node", "the Equity Desk please")[0]["value"] == "EQUITIES"
    assert not list((tmp_path / "k").rglob("*.tmp"))
