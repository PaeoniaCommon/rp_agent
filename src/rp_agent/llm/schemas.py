"""Pydantic schemas for structured LLM output (SPEC.md §5.3, §5.4, §5.6, §8.3).

The view and parameter names are enums built from the rpdata catalogue, so the
model cannot return a view or parameter that does not exist.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

Confidence = Literal["high", "medium", "low"]
Source = Literal["user_explicit", "user_implied", "note", "default", "guess"]
Scalar = str | int | float


@cache
def view_choice_model(view_names: tuple[str, ...]) -> type[BaseModel]:
    view_enum = Literal[view_names]  # type: ignore[valid-type]
    return create_model(
        "ViewChoice",
        single_request=(bool, Field(
            description="True if one get_data call on one view can meet the whole request.")),
        parts=(list[str], Field(
            description="If single_request is false, the separate data requests in it.")),
        view=(view_enum | None, Field(description="The chosen view, or null if unsure.")),
        confidence=(Confidence, ...),
        alternatives=(list[view_enum], Field(
            description="Other plausible views. Empty if the choice is clear.")),
        reason=(str, Field(description="One sentence.")),
        note_ids=(list[str], Field(description="IDs of notes or examples relied on.")),
    )


@cache
def param_extraction_model(param_names: tuple[str, ...]) -> type[BaseModel]:
    name_enum = Literal[param_names]  # type: ignore[valid-type]
    param_value = create_model(
        "ParamValue",
        name=(name_enum, ...),
        value=(Scalar | None, Field(
            description="null only for a mandatory parameter with no value in the request.")),
        source=(Source, ...),
        note_ids=(list[str], Field(description="IDs of notes relied on for this value.")),
        confidence=(Confidence, ...),
    )
    return create_model(
        "ParamExtraction",
        params=(list[param_value], ...),
        questions=(list[str], Field(description="Anything you are unsure about.")),
    )


class ReviewEdit(BaseModel):
    name: str = Field(description="A parameter name, or 'view' to change the view.")
    value: Scalar


class ReviewDecision(BaseModel):
    action: Literal["approve", "edit", "cancel"]
    edits: list[ReviewEdit] = Field(description="Changes the reviewer asked for, if any.")


class NoteDraft(BaseModel):
    text: str = Field(description="One short sentence, at most 200 characters.")


class PhraseDraft(BaseModel):
    phrase: str | None = Field(
        description="The exact words from the request (at most 5), or null if none fit.")
