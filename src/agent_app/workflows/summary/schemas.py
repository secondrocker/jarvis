"""Schemas and state for the structured summary workflow."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class SummaryInput(BaseModel):
    """Validated options accepted by the summary workflow."""

    text: str = Field(min_length=1)
    language: str | None = Field(default=None, max_length=32)
    max_words: int = Field(default=200, ge=50, le=1000)

    @field_validator("text")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        """Reject text that contains no content after outer whitespace is removed."""
        text = value.strip()
        if not text:
            raise ValueError("text must not be blank")
        return text


class SummaryResult(BaseModel):
    """Structured response produced by the summary model."""

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1, max_length=10)


class SummaryState(TypedDict, total=False):
    """State passed through the fixed summary graph."""

    text: str
    language: str | None
    max_words: int
    normalized_text: str
    result: dict[str, Any]
