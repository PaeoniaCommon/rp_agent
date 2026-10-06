# rp_agent

A LangGraph agent that turns one plain-language data request into **one**
correct, economical `rpdata.get_data` call. Results are kept in a per-user
in-memory store and returned as a dataset ID with the exact view, parameters
and retrieval time. The agent learns notes about views and parameters as it
runs, so over time it needs fewer human reviews and makes fewer bad calls.

See [`SPEC.md`](SPEC.md) for the full specification.

## Install

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...        # the default model is claude-opus-5-5
```

`rpdata` is installed from `git+https://github.com/PaeoniaCommon/rpdata.git`.
For local work on both repos, install rpdata first with `pip install -e ../rpdata`.

## Quick start

```python
from rp_agent import RPAgent

agent = RPAgent()
config = {"configurable": {"user_id": "U004512", "thread_id": "t1"}}

reply = agent.chat("Tier1 breaches on the equity desk on 28 Aug 2026", config=config)
print(reply.text)
if reply.waiting_for_review:        # only when something is uncertain
    reply = agent.chat("node EQUITIES", config=config)

df = agent.store.get_df("U004512", reply.dataset["dataset_id"])
agent.close()                       # drain background learning
```

Or interactively:

```bash
rp-agent chat --user-id U004512
```

## How it works

- **One request, one view, one call.** `get_data` is not an LLM tool. The only
  path to it is the `fetch` node, behind validation, the cache and the budget.
  The agent never retries by itself. One more call is allowed only after a failed
  call **and** a human approving a corrected proposal, so a request makes at most two.
- **Validation before the call.** Every parameter is checked with rpdata's own
  `Parameter.validate`. Safe fixes (case, whitespace, unambiguous date formats)
  are applied and reported. Anything uncertain goes to human review.
- **Human review only when uncertain.** When the agent is sure, it calls straight away.
- **30-minute reuse per user.** If the same user asks for the same view and
  parameters within 30 minutes, from any conversation, the existing dataset ID
  is returned and no call is made.
- **Learning in the background.** Nodes emit events onto a queue, and a
  worker thread writes notes to `knowledge/` (git-ignored). The reply never waits
  for learning.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `RP_AGENT_MODEL` | `claude-opus-5-5` | Main model |
| `RP_AGENT_EFFORT` | `medium` | Effort for the main model |
| `RP_AGENT_LEARN_MODEL` | same as `RP_AGENT_MODEL` | Model for phrasing notes in the background |
| `RP_AGENT_KNOWLEDGE_DIR` | `./knowledge` | Learned notes and value indexes |
| `RP_AGENT_LOG_DIR` | `./logs` | `get_data_calls.jsonl` |
| `RP_AGENT_PARAM_DOMAINS` | | Override value domains, e.g. `node=closed_unpublished` |

## Development

```bash
pytest                      # uses a scripted fake model; no network or API key
ruff check .
python scripts/eval_learning.py   # learning curve with the real model (costs API calls)
```
