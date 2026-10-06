"""Learning-curve evaluation (SPEC.md §11). Needs a real LLM (ANTHROPIC_API_KEY); run by hand.

Runs a fixed set of single data requests twice: first with an empty knowledge
directory, then again with the notes the first run produced. For each run it
reports human reviews per request, get_data calls per request, failed calls and
cache hits. The second run should need fewer reviews and make fewer failed calls.

    python scripts/eval_learning.py [--knowledge-dir DIR]

Every request uses its own user_id, so the 30-minute reuse never hides a call.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import rpdata

from rp_agent import RPAgent, Settings

# (request, scripted answers to any human review, in order)
REQUESTS: list[tuple[str, list[str]]] = [
    ("Tier1 limits for the equity desk", ["node EQUITIES"]),
    ("Show me the equity desk limits at Warning level", ["node EQUITIES"]),
    ("Utilisation for the equity desk on 2026-07-01", ["node EQUITIES"]),
    ("tier1 breaches on the equity desk on 1 July 2026", ["node EQUITIES"]),
    ("all limits", ["max_depth 1"]),
    ("every limit across the firm", ["max_depth 1"]),
    ("credit limits", ["ok"]),
    ("limits for credit high yield", ["node CR_HY"]),
    ("CR_HY limits", ["ok"]),
    ("FX spot utilisation on 2026/07/02", ["ok"]),
    ("FX spot utilisation on 2026-07-03", ["ok"]),
    ("which users are in LG_FX", ["ok"]),
    ("members of the senior management limit group", ["limit_group LG_SENIOR_MGMT"]),
    ("members of the senior management group", ["limit_group LG_SENIOR_MGMT"]),
    ("node tree under EQUITIES", ["ok"]),
    ("the hierarchy below commodities", ["node COMMODITIES"]),
    ("EQ Delta limits on EQUITIES", ["ok"]),
    ("eq delta limits on equities", ["ok"]),
    ("rates G10 utilisation between 2026-07-01 and 2026-07-10", ["node RATES_G10"]),
    ("utilisation over 90% for rates g10 on 2026-07-06", ["node RATES_G10"]),
]


@dataclass
class RunStats:
    requests: int = 0
    reviews: int = 0
    calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        n = max(self.requests, 1)
        return {
            "requests": self.requests,
            "reviews_per_request": round(self.reviews / n, 2),
            "get_data_calls_per_request": round(self.calls / n, 2),
            "failed_calls": self.failed_calls,
            "cache_hits": self.cache_hits,
            "outcomes": self.outcomes,
        }


def run_once(knowledge_dir: Path, log_dir: Path, run: int) -> RunStats:
    calls: list[tuple[str, dict, bool]] = []

    def counting_get_data(view, **params):
        try:
            df = rpdata.get_data(view, **params)
        except Exception:
            calls.append((view, params, False))
            raise
        calls.append((view, params, True))
        return df

    agent = RPAgent(Settings.from_env(knowledge_dir=knowledge_dir, log_dir=log_dir),
                    get_data=counting_get_data)
    stats = RunStats()
    try:
        for i, (request, answers) in enumerate(REQUESTS):
            config = {"configurable": {"user_id": f"EVAL{run}{i:03d}", "thread_id": "t"}}
            stats.requests += 1
            reply = agent.chat(request, config)
            pending = list(answers)
            while reply.waiting_for_review:
                stats.reviews += 1
                reply = agent.chat(pending.pop(0) if pending else "cancel", config)
            stats.outcomes[reply.outcome] = stats.outcomes.get(reply.outcome, 0) + 1
            if reply.outcome == "reused":
                stats.cache_hits += 1
            print(f"[run {run}] {request!r}: {reply.outcome}")
        agent.flush()
    finally:
        agent.close()
    stats.calls = len(calls)
    stats.failed_calls = sum(1 for *_, ok in calls if not ok)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-dir", type=Path, default=None)
    args = parser.parse_args()
    base = Path(tempfile.mkdtemp(prefix="rp_agent_eval_"))
    knowledge = args.knowledge_dir or base / "knowledge"
    first = run_once(knowledge, base / "logs", 1)
    second = run_once(knowledge, base / "logs", 2)
    print(json.dumps({"run_1_empty_knowledge": first.summary(),
                      "run_2_with_notes": second.summary(),
                      "knowledge_dir": str(knowledge)}, indent=2))


if __name__ == "__main__":
    main()
