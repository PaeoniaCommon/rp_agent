"""Background learning (SPEC.md §8.3, §8.6).

Graph nodes call `Learning.emit(event)`, which puts a small immutable event on
a bounded queue with `put_nowait` and returns at once. A single daemon thread
(`LearningWorker`) does all learning work: LLM phrasing, dedup, merging,
promotion and demotion, value-domain changes and value-index updates.
The user's reply never waits for any of it.
"""

from __future__ import annotations

import atexit
import copy
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any

from . import prompts
from .knowledge import KnowledgeBase
from .llm import structured
from .schemas import NoteDraft, PhraseDraft
from .values import CLOSED_UNPUBLISHED, ValueDomains, squash

log = logging.getLogger(__name__)

# Event kinds (SPEC.md §8.3)
NORMALISED = "normalised"
DOMAIN_CLOSED = "domain_closed"
VALUES_VALIDATED = "values_validated"
REVIEW = "review"
CALL_OK = "call_ok"
CALL_ERROR = "call_error"

_FORBIDDEN_PAYLOAD_KEYS = {"user_id", "df", "dataframe"}


@dataclass(frozen=True)
class LearningEvent:
    """What the worker needs and nothing more. Never a DataFrame or a user_id."""

    kind: str
    payload: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    created: float = field(default_factory=time.time)

    @classmethod
    def make(cls, kind: str, **payload: Any) -> LearningEvent:
        bad = _FORBIDDEN_PAYLOAD_KEYS & set(payload)
        if bad:
            raise ValueError(f"LearningEvent may not carry {sorted(bad)}")
        return cls(kind=kind, payload=MappingProxyType(copy.deepcopy(payload)))


_STOP = object()


class Learning:
    """The emitting side (used on the request path) plus the worker's lifecycle."""

    def __init__(self, worker: LearningWorker, queue_size: int, shutdown_timeout: float) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker = worker
        self._shutdown_timeout = shutdown_timeout
        self._closed = False
        self.dropped = 0
        self.processed = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="rp-agent-learning", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ---------------------------------------------------------- request path

    def emit(self, event: LearningEvent) -> None:
        """Never blocks, never calls the LLM, never touches the disk."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            log.warning("Learning queue full; dropped %s event", event.kind)

    # ------------------------------------------------------------ background

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is _STOP:
                    return
                self._worker.handle(event)
                self.processed += 1
            except Exception:
                self.failed += 1
                log.exception("Learning worker failed on %s event; dropped",
                              getattr(event, "kind", event))
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------- lifecycle

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every queued event is processed. Never used on the request path."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def close(self) -> None:
        """Stop taking events, drain the queue (up to the timeout) and stop the worker."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(_STOP, timeout=self._shutdown_timeout)
        except queue.Full:
            log.warning("Learning queue still full at shutdown; events dropped")
        self._thread.join(self._shutdown_timeout)
        if self._thread.is_alive():
            log.warning("Learning worker did not finish within %ss; %d events dropped",
                        self._shutdown_timeout, self._queue.qsize())
        atexit.unregister(self.close)


class LearningWorker:
    """Turns learning events into notes and value-index updates (SPEC.md §8.3)."""

    def __init__(self, knowledge: KnowledgeBase, domains: ValueDomains, catalogue,
                 llm: Any | None = None) -> None:
        self.kb = knowledge
        self.domains = domains
        self.catalogue = catalogue
        self.llm = llm

    def handle(self, event: LearningEvent) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is None:
            log.warning("Unknown learning event %s", event.kind)
            return
        handler(dict(event.payload))

    # ------------------------------------------------------------ phrasing

    def _phrase(self, fallback: str, facts: str) -> str:
        """LLM-phrased note text, checked deterministically; falls back to `fallback`."""
        if self.llm is None:
            return fallback
        try:
            draft = structured(self.llm, NoteDraft, prompts.NOTE_SYSTEM, facts)
            text = " ".join(draft.text.split())
        except Exception as exc:  # noqa: BLE001 - phrasing is optional; use the fallback
            log.warning("Note phrasing failed (%s); using fallback text", exc)
            return fallback
        if not text or len(text) > 300:
            return fallback
        return text

    def _phrase_from_request(self, request: str, target: str) -> str | None:
        """The words in `request` that refer to `target`, verified to be in the request."""
        if self.llm is None or not request:
            return None
        try:
            draft = structured(self.llm, PhraseDraft, prompts.PHRASE_SYSTEM,
                               prompts.phrase_user(request, target))
        except Exception as exc:  # noqa: BLE001 - no phrase means no alias
            log.warning("Phrase extraction failed (%s)", exc)
            return None
        phrase = (draft.phrase or "").strip().strip("'\"").strip()
        if not phrase or len(phrase.split()) > 5 or phrase.lower() not in request.lower():
            return None
        return phrase

    def _is_date_param(self, param: str) -> bool:
        p = self.catalogue.params.get(param)
        return bool(p and p.format and p.format.startswith("yyyy-mm-dd"))

    # -------------------------------------------------------------- events

    def _on_normalised(self, e: dict[str, Any]) -> None:
        param, reason = e["param"], e["reason"]
        fallback = (f"{param}: {e['from']!r} was rejected ({e['message']}); "
                    f"{e['to']!r} passes. Use this form.")
        text = self._phrase(fallback, prompts.note_facts(
            "A parameter value failed rpdata validation and a corrected form passed.", e))
        self.kb.add_rule("param", param, "normalisation", text, "certain", "rpdata_error",
                         data={"reason": reason, "example_from": e["from"],
                               "example_to": e["to"]},
                         dedup_key=f"normalisation:{param}:{reason}")
        if self.domains.domain(param) == CLOSED_UNPUBLISHED and isinstance(e["to"], str):
            index = self.domains.index(param)
            index.add(e["to"])
            index.save()

    def _on_domain_closed(self, e: dict[str, Any]) -> None:
        param = e["param"]
        self.kb.set_domain(param, CLOSED_UNPUBLISHED)
        self.kb.add_rule(
            "param", param, "value_domain",
            f"Closed list that rpdata does not publish: unknown but well-formed values are "
            f"rejected ({e['message']}).",
            "certain", "rpdata_error", dedup_key=f"value_domain:{param}")

    def _on_values_validated(self, e: dict[str, Any]) -> None:
        touched = set()
        for param, value in e["values"].items():
            if isinstance(value, str) and self.domains.domain(param) == CLOSED_UNPUBLISHED:
                self.domains.index(param).add(value)
                touched.add(param)
        for param in e.get("rejected", []):
            # The reference data has changed: remove a value rpdata now rejects.
            name, value = param
            if self.domains.domain(name) == CLOSED_UNPUBLISHED:
                self.domains.index(name).discard(value)
                touched.add(name)
        for param in touched:
            self.domains.index(param).save()

    def _on_review(self, e: dict[str, Any]) -> None:
        request = e["request"]
        note_ids_by_param: dict[str, list[str]] = e.get("note_ids_by_param", {})
        for param in e.get("edited", []):
            self.kb.demote_notes(note_ids_by_param.get(param, []))
            self._learn_alias(request, param, e["params_after"].get(param), "likely")
        for param in e.get("approved_uncertain", []):
            self._learn_alias(request, param, e["params_after"].get(param), "tentative")
        view_after = e.get("view_after")
        if e.get("view_changed"):
            self.kb.demote_notes(e.get("view_note_ids", []))
            self._learn_view_hint(request, view_after, "likely")
        elif e.get("view_was_uncertain") and view_after:
            self._learn_view_hint(request, view_after, "tentative")

    def _learn_alias(self, request: str, param: str, value: Any, certainty: str) -> None:
        if value is None or isinstance(value, bool) or self._is_date_param(param):
            return  # relative dates ("yesterday") must never become fixed aliases
        phrase = self._phrase_from_request(request, f"{param} = {value!r}")
        if phrase is None or (isinstance(value, str) and squash(phrase) == squash(value)):
            return
        self.kb.add_alias(param, phrase, value, certainty, "human_review")

    def _learn_view_hint(self, request: str, view: str | None, certainty: str) -> None:
        if not view:
            return
        phrase = self._phrase_from_request(request, f"the data view {view}")
        if phrase is None:
            return
        self.kb.add_rule("general", "general", "view_hint",
                         f"Requests mentioning '{phrase}' -> {view}", certainty, "human_review",
                         data={"phrase": phrase, "view": view},
                         dedup_key=f"view_hint:{phrase.lower()}")

    def _on_call_ok(self, e: dict[str, Any]) -> None:
        self.kb.confirm(e.get("note_ids", []))
        view, params = e["view"], e["params"]
        if not e.get("reviewed") and not e.get("empty"):
            self.kb.add_example(view, e["request"], params)
        cov = self.kb.coverage()
        updates: dict[str, Any] = {}
        if e.get("date_range"):
            first, last = e["date_range"]
            if not cov.get("first_seen") or first < cov["first_seen"]:
                updates["first_seen"] = first
            if not cov.get("last_seen") or last > cov["last_seen"]:
                updates["last_seen"] = last
        if e.get("empty"):
            updates.update(self._empty_result_coverage(params, {**cov, **updates}))
        if updates:
            self.kb.update_general({"coverage": updates})

    def _empty_result_coverage(self, params: dict[str, Any], cov: dict) -> dict[str, Any]:
        dates = {k: v for k, v in params.items() if self._is_date_param(k) and isinstance(v, str)}
        if not dates:
            return {}
        updates: dict[str, Any] = {}
        try:
            parsed = sorted(date.fromisoformat(v) for v in dates.values())
        except ValueError:
            return {}
        start, end = parsed[0], parsed[-1]
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if days and all(d.weekday() >= 5 for d in days):
            self.kb.add_rule("general", "general", "coverage",
                             "Weekend dates return an empty DataFrame (data is business days "
                             "only).", "likely", "success", dedup_key="coverage:weekend")
        last_seen, first_seen = cov.get("last_seen"), cov.get("first_seen")
        if (last_seen and start.isoformat() > last_seen
                and (not cov.get("empty_after") or start.isoformat() < cov["empty_after"])):
            updates["empty_after"] = start.isoformat()
        if (first_seen and end.isoformat() < first_seen
                and (not cov.get("empty_before") or end.isoformat() > cov["empty_before"])):
            updates["empty_before"] = end.isoformat()
        return updates

    def _on_call_error(self, e: dict[str, Any]) -> None:
        self.kb.demote_notes(e.get("note_ids", []))
        error_class, message = e["error_class"], e["message"]
        view, params = e["view"], e["params"]
        if error_class == "ViewTimeoutError":
            self._learn_requires_filter(view, params, message)
        elif error_class == "UnknownNodeError":
            self._on_domain_closed({"param": "node", "message": message})
        elif error_class == "InvalidParameterError":
            order = re.search(r"(\w+)=\S+ must not be (?:after|greater than) (\w+)=", message)
            if order:
                low, high = order.group(1), order.group(2)
                text = self._phrase(f"{low} must be <= {high} ({message}).",
                                    prompts.note_facts("A cross-parameter rule failed.", e))
                self.kb.add_rule("general", "general", "order", text, "certain", "rpdata_error",
                                 data={"low": low, "high": high},
                                 dedup_key=f"order:{low}:{high}")
            else:
                named = re.search(r":\s*(\w+)=", message)
                if named and named.group(1) in self.catalogue.params:
                    param = named.group(1)
                    self.kb.add_rule("param", param, "format", message, "certain",
                                     "rpdata_error", dedup_key=f"format:{param}:{message}")
        # UnknownView / UnexpectedParameter / MissingParameter are bugs (§6.3): nothing to learn.

    def _learn_requires_filter(self, view: str, params: dict[str, Any], message: str) -> None:
        trigger = re.search(r"(\w+)='([^']*)'", message)
        if not trigger or trigger.group(1) not in params:
            return
        param, value = trigger.group(1), trigger.group(2)
        # Params sent alongside the trigger were not enough. Date params generalise to any
        # value; others keep the exact value seen (e.g. max_depth='max').
        insufficient = {
            k: ("*" if self._is_date_param(k) else v)
            for k, v in params.items() if k != param
        }
        existing = [r for r in self.kb.rules("view", view, "requires_filter")
                    if r.get("data", {}).get("param") == param
                    and r.get("data", {}).get("value") == value]
        if existing:
            merged = dict(existing[0]["data"].get("insufficient", {}))
            merged.update(insufficient)
            insufficient = merged
        listed = ", ".join(f"{k}={v}" for k, v in sorted(insufficient.items())) or "nothing else"
        fallback = (f"{param}={value!r} times out unless another filter is added; "
                    f"not enough on their own: {listed}. ({message})")
        text = self._phrase(fallback, prompts.note_facts(
            "A get_data call timed out because it was too broad.",
            {"view": view, "params": params, "message": message}))
        self.kb.add_rule("view", view, "requires_filter", text, "certain", "rpdata_error",
                         data={"param": param, "value": value, "insufficient": insufficient},
                         dedup_key=f"requires_filter:{param}:{value}")


def iso_dates_of(df) -> tuple[str, str] | None:
    """The first and last date in a DataFrame's `date` column, if it has one."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    col = df["date"]
    first, last = col.min(), col.max()
    if isinstance(first, (datetime, date)) or hasattr(first, "isoformat"):
        return first.isoformat()[:10], last.isoformat()[:10]
    return str(first)[:10], str(last)[:10]
