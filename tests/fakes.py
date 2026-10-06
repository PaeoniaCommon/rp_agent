"""Test doubles: a scripted chat model, a fake clock and a get_data spy."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import rpdata


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class GetDataSpy:
    """Wraps rpdata.get_data and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, view, **params):
        self.calls.append((view, dict(params)))
        return rpdata.get_data(view, **params)


class _Bound:
    def __init__(self, llm: FakeLLM, schema) -> None:
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        name = self.schema.__name__
        user = messages[-1].content
        self.llm.calls.append((name, user))
        handler = self.llm.handlers.get(name)
        if handler is None:
            raise RuntimeError(f"FakeLLM has no handler for {name}")
        if isinstance(handler, list):
            if not handler:
                raise RuntimeError(f"FakeLLM ran out of scripted {name} answers")
            handler = handler.pop(0)
        result = handler(user) if callable(handler) else handler
        return self.schema.model_validate(result)


class FakeLLM:
    """`with_structured_output(schema).invoke(messages)` answered from `handlers`.

    `handlers` maps a schema name (ViewChoice, ParamExtraction, ReviewDecision,
    NoteDraft, PhraseDraft) to a dict, a callable(user_prompt) -> dict, or a list of
    either (consumed in order).
    """

    def __init__(self, **handlers: Any) -> None:
        self.handlers: dict[str, Any] = handlers
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema, method: str = "json_schema"):
        assert method == "json_schema"
        return _Bound(self, schema)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


class BlockingLLM(FakeLLM):
    """A learning model that blocks until released (to prove learning is off the request path)."""

    def __init__(self, **handlers: Any) -> None:
        super().__init__(**handlers)
        self.release = threading.Event()
        self.entered = threading.Event()

    def with_structured_output(self, schema, method: str = "json_schema"):
        bound = super().with_structured_output(schema, method)
        outer = self

        class _Blocking:
            def invoke(self, messages):
                outer.entered.set()
                outer.release.wait(30)
                return bound.invoke(messages)

        return _Blocking()


def view(name: str | None, confidence: str = "high", alternatives=(), single: bool = True,
         parts=(), note_ids=()) -> dict:
    return {"single_request": single, "parts": list(parts), "view": name,
            "confidence": confidence, "alternatives": list(alternatives),
            "reason": "test", "note_ids": list(note_ids)}


def params(questions=(), **values: Any) -> dict:
    """params(node="EQUITIES", limit_level=("Tier1", "user_implied", "medium"))."""
    out = []
    for name, spec in values.items():
        if isinstance(spec, tuple):
            value, source, confidence = (list(spec) + ["user_explicit", "high"])[:3]
        else:
            value, source, confidence = spec, "user_explicit", "high"
        if value is None:
            source, confidence = "guess", "low"
        out.append({"name": name, "value": value, "source": source, "note_ids": [],
                    "confidence": confidence})
    return {"params": out, "questions": list(questions)}


Responder = Callable[[str], dict]
