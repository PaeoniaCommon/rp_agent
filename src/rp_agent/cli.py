"""Interactive CLI: `rp-agent chat --user-id U004512` (SPEC.md §9.2)."""

from __future__ import annotations

import argparse
import logging
import uuid

import pandas as pd

from .agent import RPAgent

HELP = """Commands:
  /datasets      list your stored datasets
  /show <id>     show a dataset's metadata and first rows
  /refresh       the next request skips the 30-minute reuse
  /quit          exit"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rp-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="interactive data requests")
    chat.add_argument("--user-id", required=True)
    chat.add_argument("--thread-id", default=None)
    chat.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    agent = RPAgent()
    config = {"configurable": {"user_id": args.user_id,
                               "thread_id": args.thread_id or uuid.uuid4().hex[:8]}}
    refresh_next = False
    print(f"rp-agent — user {args.user_id}. Type a data request, or /help.")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                print(HELP)
                continue
            if line == "/datasets":
                for r in agent.store.list(args.user_id):
                    meta = r.metadata()
                    print(f"{meta['dataset_id']}  {meta['retrieved_at']}  {meta['view']}  "
                          f"{meta['params']}  ({meta['row_count']} rows)")
                continue
            if line.startswith("/show"):
                parts = line.split()
                if len(parts) != 2:
                    print("usage: /show <dataset_id>")
                    continue
                try:
                    record = agent.store.get(args.user_id, parts[1])
                except KeyError as exc:
                    print(exc)
                    continue
                print(record.metadata())
                with pd.option_context("display.width", 160, "display.max_columns", 20):
                    print(record.df.head(10))
                continue
            if line == "/refresh":
                refresh_next = True
                print("The next request will skip the 30-minute reuse.")
                continue
            reply = agent.chat(line, config, refresh=refresh_next)
            if not reply.waiting_for_review:
                refresh_next = False
            print(reply.text)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
