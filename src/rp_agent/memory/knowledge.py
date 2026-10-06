"""KnowledgeBase: learned notes on views and parameters (SPEC.md §8).

Notes are Markdown files with a YAML front matter block:

    knowledge/general.md
    knowledge/views/<View>.md
    knowledge/params/<param>.md

The front matter holds machine-usable facts (rules, aliases, worked examples,
value domain); the body is free text for people and the LLM. Lists of values
are never kept here: see `rp_agent.memory.values` (SPEC.md §8.4).

Only the learning worker writes (SPEC.md §8.6). Requests read from an
in-memory copy guarded by a lock, and every file write is atomic.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

CERTAINTY_ORDER = ("tentative", "likely", "certain")
CONFIRMATIONS_TO_PROMOTE = 2

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _today(clock: Callable[[], datetime]) -> str:
    return clock().date().isoformat()


def promote(certainty: str) -> str:
    i = CERTAINTY_ORDER.index(certainty)
    return CERTAINTY_ORDER[min(i + 1, len(CERTAINTY_ORDER) - 1)]


def demote(certainty: str) -> str | None:
    """One level down, or None (remove) below tentative."""
    i = CERTAINTY_ORDER.index(certainty)
    return CERTAINTY_ORDER[i - 1] if i > 0 else None


class KnowledgeBase:
    def __init__(
        self,
        root: Path,
        clock: Callable[[], datetime],
        max_rules_per_file: int = 25,
        max_examples_per_view: int = 10,
        max_aliases: int = 500,
    ) -> None:
        self.root = Path(root)
        self.clock = clock
        self.max_rules_per_file = max_rules_per_file
        self.max_examples_per_view = max_examples_per_view
        self.max_aliases = max_aliases
        self._lock = threading.RLock()
        # (kind, name) -> {"front": dict, "body": str}
        self._docs: dict[tuple[str, str], dict[str, Any]] = {}
        self.root.mkdir(parents=True, exist_ok=True)  # created empty on first run (§8.2)
        self._load()

    # ------------------------------------------------------------------ files

    def _path(self, kind: str, name: str) -> Path:
        if kind == "general":
            return self.root / "general.md"
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(f"Unsafe knowledge file name: {name!r}")
        return self.root / f"{kind}s" / f"{name}.md"

    def _load(self) -> None:
        for kind, pattern in (("general", "general.md"), ("view", "views/*.md"),
                              ("param", "params/*.md")):
            for path in self.root.glob(pattern):
                name = "general" if kind == "general" else path.stem
                text = path.read_text(encoding="utf-8")
                match = _FRONT_MATTER_RE.match(text)
                if match:
                    front = yaml.safe_load(match.group(1)) or {}
                    body = match.group(2)
                else:
                    front, body = {}, text
                self._docs[(kind, name)] = {"front": front, "body": body}

    def _doc(self, kind: str, name: str) -> dict[str, Any]:
        key = (kind, name)
        if key not in self._docs:
            front: dict[str, Any] = {}
            if kind in ("view", "param"):
                front[kind] = name
            self._docs[key] = {"front": front, "body": ""}
        return self._docs[key]

    def _save(self, kind: str, name: str) -> None:
        doc = self._docs[(kind, name)]
        path = self._path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "---\n"
            + yaml.safe_dump(doc["front"], sort_keys=False, allow_unicode=True).rstrip("\n")
            + "\n---\n"
            + doc["body"]
        )
        # Atomic: a reader sees the old file or the new one, never half of one.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ reads

    def front(self, kind: str, name: str) -> dict[str, Any]:
        with self._lock:
            doc = self._docs.get((kind, name))
            return copy.deepcopy(doc["front"]) if doc else {}

    def body(self, kind: str, name: str) -> str:
        with self._lock:
            doc = self._docs.get((kind, name))
            return doc["body"] if doc else ""

    def rules(self, kind: str, name: str, rule_kind: str | None = None) -> list[dict[str, Any]]:
        rules = self.front(kind, name).get("rules", [])
        return [r for r in rules if rule_kind is None or r.get("kind") == rule_kind]

    def all_rules(self, rule_kind: str) -> list[tuple[str, str, dict[str, Any]]]:
        with self._lock:
            out = []
            for (kind, name), doc in self._docs.items():
                for r in doc["front"].get("rules", []):
                    if r.get("kind") == rule_kind:
                        out.append((kind, name, copy.deepcopy(r)))
            return out

    def aliases(self, param: str) -> list[dict[str, Any]]:
        return self.front("param", param).get("aliases", [])

    def aliases_in_text(self, param: str, text: str) -> list[dict[str, Any]]:
        """Only aliases whose phrase appears in the request are loaded (§8.4)."""
        lowered = text.lower()
        return [a for a in self.aliases(param) if str(a.get("phrase", "")).lower() in lowered]

    def examples(self, view: str) -> list[dict[str, Any]]:
        return self.front("view", view).get("examples", [])

    def value_domain(self, param: str) -> str | None:
        return self.front("param", param).get("value_domain")

    def defaults(self) -> dict[str, Any]:
        """Certain `default_value`s, used as cache equivalences (§7.3)."""
        with self._lock:
            out = {}
            for (kind, name), doc in self._docs.items():
                front = doc["front"]
                if (kind == "param" and "default_value" in front
                        and front.get("default_certainty", "certain") == "certain"):
                    out[name] = front["default_value"]
            return out

    def coverage(self) -> dict[str, Any]:
        return self.front("general", "general").get("coverage", {})

    def date_convention(self) -> str | None:
        """'dmy' or 'mdy' if a certain note records the users' convention."""
        front = self.front("general", "general")
        if front.get("date_convention_certainty") == "certain":
            return front.get("date_convention")
        return None

    def note_location(self, note_id: str) -> tuple[str, str] | None:
        with self._lock:
            for (kind, name), doc in self._docs.items():
                front = doc["front"]
                for item in front.get("rules", []) + front.get("aliases", []):
                    if item.get("id") == note_id:
                        return kind, name
            return None

    def find_note(self, note_id: str) -> dict[str, Any] | None:
        with self._lock:
            for doc in self._docs.values():
                front = doc["front"]
                for item in front.get("rules", []) + front.get("aliases", []):
                    if item.get("id") == note_id:
                        return copy.deepcopy(item)
            return None

    # ----------------------------------------------------------------- writes
    # Called only by the learning worker (SPEC.md §8.6).

    def _next_id(self, prefix: str) -> str:
        used = set()
        for doc in self._docs.values():
            for item in doc["front"].get("rules", []) + doc["front"].get("aliases", []):
                used.add(item.get("id"))
        n = 1
        while f"{prefix}-{n:03d}" in used:
            n += 1
        return f"{prefix}-{n:03d}"

    def add_rule(
        self,
        kind: str,
        name: str,
        rule_kind: str,
        text: str,
        certainty: str,
        source: str,
        data: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> str:
        """Add a rule, or refresh the existing one with the same `dedup_key`.

        A rule that already exists keeps the higher certainty and is confirmed.
        """
        if certainty not in CERTAINTY_ORDER:
            raise ValueError(f"Bad certainty {certainty!r}")
        with self._lock:
            doc = self._doc(kind, name)
            rules = doc["front"].setdefault("rules", [])
            today = _today(self.clock)
            key = dedup_key or f"{rule_kind}:{text}"
            for rule in rules:
                if rule.get("dedup_key") == key:
                    if CERTAINTY_ORDER.index(certainty) > CERTAINTY_ORDER.index(rule["certainty"]):
                        rule["certainty"] = certainty
                    rule["text"] = text
                    if data is not None:
                        rule["data"] = data
                    rule["last_confirmed"] = today
                    self._save(kind, name)
                    return rule["id"]
            rule_id = self._next_id(name)
            rules.append({
                "id": rule_id,
                "kind": rule_kind,
                "text": text,
                "certainty": certainty,
                "source": source,
                "created": today,
                "last_confirmed": today,
                "uses": 0,
                "confirmations": 0,
                "dedup_key": key,
                **({"data": data} if data is not None else {}),
            })
            self._enforce_rule_limit(rules)
            self._save(kind, name)
            return rule_id

    def _enforce_rule_limit(self, rules: list[dict[str, Any]]) -> None:
        # Drop the least-used, least-certain rules first; never human-sourced ones.
        while len(rules) > self.max_rules_per_file:
            candidates = [r for r in rules if r.get("source") != "human"]
            if not candidates:
                return
            victim = min(
                candidates,
                key=lambda r: (CERTAINTY_ORDER.index(r["certainty"]), r.get("uses", 0),
                               r.get("last_confirmed", "")),
            )
            rules.remove(victim)

    def add_alias(self, param: str, phrase: str, value: Any, certainty: str, source: str) -> str:
        with self._lock:
            doc = self._doc("param", param)
            aliases = doc["front"].setdefault("aliases", [])
            today = _today(self.clock)
            for alias in aliases:
                if str(alias["phrase"]).lower() == phrase.lower():
                    if alias["value"] != value:
                        # The newest human answer wins; start again from this certainty.
                        alias["value"] = value
                        alias["certainty"] = certainty
                        alias["confirmations"] = 0
                    elif CERTAINTY_ORDER.index(certainty) > CERTAINTY_ORDER.index(
                        alias["certainty"]
                    ):
                        alias["certainty"] = certainty
                    alias["last_used"] = today
                    self._save("param", param)
                    return alias["id"]
            alias_id = self._next_id(f"{param}-alias")
            aliases.append({
                "id": alias_id,
                "phrase": phrase,
                "value": value,
                "certainty": certainty,
                "source": source,
                "uses": 0,
                "confirmations": 0,
                "last_used": today,
            })
            # Least recently used are dropped first (§8.4).
            if len(aliases) > self.max_aliases:
                aliases.sort(key=lambda a: (a.get("last_used", ""), a.get("uses", 0)))
                del aliases[: len(aliases) - self.max_aliases]
            self._save("param", param)
            return alias_id

    def add_example(self, view: str, request: str, params: dict[str, Any]) -> None:
        with self._lock:
            doc = self._doc("view", view)
            examples = doc["front"].setdefault("examples", [])
            examples[:] = [e for e in examples if e.get("request") != request]
            examples.insert(0, {"request": request, "params": params,
                                "added": _today(self.clock)})
            del examples[self.max_examples_per_view:]
            self._save("view", view)

    def set_domain(self, param: str, domain: str) -> None:
        with self._lock:
            doc = self._doc("param", param)
            if doc["front"].get("value_domain") != domain:
                doc["front"]["value_domain"] = domain
                self._save("param", param)

    def update_general(self, updates: dict[str, Any]) -> None:
        with self._lock:
            doc = self._doc("general", "general")
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(doc["front"].get(key), dict):
                    doc["front"][key].update(value)
                else:
                    doc["front"][key] = value
            self._save("general", "general")

    def confirm(self, note_ids: Iterable[str]) -> None:
        """A call that relied on these notes succeeded with no human edit (§8.3)."""
        self._touch(note_ids, success=True)

    def demote_notes(self, note_ids: Iterable[str]) -> None:
        """A call failed, or a reviewer overrode a value, although a note was followed."""
        self._touch(note_ids, success=False)

    def _touch(self, note_ids: Iterable[str], success: bool) -> None:
        ids = set(note_ids)
        if not ids:
            return
        with self._lock:
            today = _today(self.clock)
            for (kind, name), doc in self._docs.items():
                changed = False
                for field in ("rules", "aliases"):
                    items = doc["front"].get(field, [])
                    for item in list(items):
                        if item.get("id") not in ids:
                            continue
                        changed = True
                        if success:
                            item["uses"] = item.get("uses", 0) + 1
                            item["confirmations"] = item.get("confirmations", 0) + 1
                            item["last_confirmed" if field == "rules" else "last_used"] = today
                            if item["confirmations"] >= CONFIRMATIONS_TO_PROMOTE:
                                item["certainty"] = promote(item["certainty"])
                                item["confirmations"] = 0
                        elif item.get("source") != "human":
                            lower = demote(item["certainty"])
                            if lower is None:
                                items.remove(item)
                            else:
                                item["certainty"] = lower
                                item["confirmations"] = 0
                if changed:
                    self._save(kind, name)

