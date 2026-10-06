"""Prompt text and the compact context blocks shown to the LLM.

Only metadata reaches the model: never DataFrame rows (SPEC.md §3, decision 5).
Only the notes relevant to the current request are loaded (SPEC.md §8.2), and
long value lists appear only as a shortlist (SPEC.md §8.4).
"""

from __future__ import annotations

import json
from typing import Any

VIEW_SYSTEM = """\
You route data requests for a market-risk data service (rpdata). Each request must be
met by exactly ONE get_data call on ONE view. Calls are expensive, so precision matters.

Decide:
- single_request: false if the request needs more than one view or more than one call
  (for example "limits AND utilisations"). List the separate requests in `parts`.
- view: the single view that answers the request, chosen ONLY from the listed views.
  Use null if no listed view fits.
- confidence: "high" only if one view clearly fits (an explicit view name, a matching
  note or worked example, or an unmistakable description). Otherwise "medium" or "low",
  and put every plausible view in `alternatives`.
- note_ids: the IDs of any notes or examples you relied on.
Do not invent views. Do not guess: lower confidence instead."""

PARAMS_SYSTEM = """\
You fill in parameters for ONE get_data call on a market-risk data service (rpdata).
Calls are expensive: a wrong value wastes a call, so never guess silently.

Rules:
- Return an entry for every MANDATORY parameter. If the request gives no value, set
  value to null, source "guess", confidence "low".
- Set OPTIONAL parameters only when the request asks for that filter. Never add filters
  the user did not ask for.
- source: "user_explicit" (the user wrote the value), "user_implied" (clearly implied,
  e.g. "breaches" -> min_utilisation 100), "note" (taken from a note or alias below),
  "default", or "guess".
- confidence "high" only when you are sure the value is what the user means.
- Resolve relative dates ("yesterday", "last Friday") to explicit yyyy-mm-dd strings using
  today's date. Dates must be strings in yyyy-mm-dd form.
- Use the exact spelling and case shown in value lists and notes.
- note_ids: the IDs of notes/aliases you relied on for that value.
- Put anything you are unsure about in `questions`."""

REVIEW_SYSTEM = """\
A user is replying to a review of a proposed data query. Classify the reply:
- "approve": they accept the proposal as shown.
- "cancel": they do not want the query.
- "edit": they give corrections. Put each change in `edits` as a parameter name (or
  "view") and its new value, using exact parameter names from the proposal.
Copy values exactly as the user gave them."""

NOTE_SYSTEM = """\
Write ONE short sentence (at most 200 characters) for an agent's notes about how to query
a data service correctly next time. State the rule plainly. Do not include any data
values from query results and do not mention any user."""

PHRASE_SYSTEM = """\
Find the words in the user's request that refer to the given target. Return them exactly
as written in the request (at most 5 words), or null if no words clearly refer to it."""


def _rules_block(rules: list[dict[str, Any]]) -> str:
    lines = []
    for r in rules:
        lines.append(f"  - [{r['id']}] ({r['certainty']}) {r['text']}")
    return "\n".join(lines)


def view_user(request: str, catalogue_text: str, general_rules: list[dict],
              view_rules: dict[str, list[dict]], examples: dict[str, list[dict]],
              today: str) -> str:
    parts = [f"Today: {today}", "", "Views:", catalogue_text]
    if general_rules:
        parts += ["", "General notes:", _rules_block(general_rules)]
    for view, rules in view_rules.items():
        if rules:
            parts += ["", f"Notes on {view}:", _rules_block(rules)]
    example_lines = []
    for view, exs in examples.items():
        for i, ex in enumerate(exs):
            example_lines.append(f"  - [{view}-example-{i + 1}] {ex['request']!r} -> {view}")
    if example_lines:
        parts += ["", "Worked examples:", *example_lines]
    parts += ["", f"Request: {request}"]
    return "\n".join(parts)


def params_user(request: str, view, param_blocks: list[str], general_rules: list[dict],
                view_rules: list[dict], examples: list[dict], today: str) -> str:
    parts = [f"Today: {today}", "", f"View: {view.name} — {view.description}", "",
             "Parameters:", *param_blocks]
    if view_rules:
        parts += ["", f"Notes on {view.name}:", _rules_block(view_rules)]
    if general_rules:
        parts += ["", "General notes:", _rules_block(general_rules)]
    if examples:
        parts += ["", "Worked examples (requests that succeeded):"]
        for ex in examples:
            parts.append(f"  - {ex['request']!r} -> {json.dumps(ex['params'], default=str)}")
    parts += ["", f"Request: {request}"]
    return "\n".join(parts)


def param_block(name: str, mandatory: bool, parameter, domain: str, values_text: str,
                aliases: list[dict], rules: list[dict]) -> str:
    lines = [
        (f"- {name} ({'MANDATORY' if mandatory else 'optional'}; type {parameter.dtype}): "
         f"{parameter.description}"),
    ]
    if parameter.format:
        lines.append(f"    format: {parameter.format}")
    if values_text:
        lines.append(f"    values: {values_text}")
    for a in aliases:
        lines.append(f"    alias [{a['id']}] ({a['certainty']}): "
                     f"{a['phrase']!r} means {a['value']!r}")
    if rules:
        lines.append("    notes:")
        lines.append(_rules_block(rules).replace("  - ", "      - "))
    return "\n".join(lines)


def review_user(review_text: str, reply: str) -> str:
    return f"Proposal shown to the user:\n{review_text}\n\nUser's reply: {reply}"


def note_facts(summary: str, facts: dict[str, Any]) -> str:
    return f"{summary}\nFacts:\n{json.dumps(dict(facts), default=str, indent=2)}"


def phrase_user(request: str, target: str) -> str:
    return f"Request: {request}\nTarget: {target}"
