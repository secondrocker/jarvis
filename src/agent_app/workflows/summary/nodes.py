"""Nodes used by the structured summary graph."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.workflows.summary.prompts import SUMMARY_PROMPT
from agent_app.workflows.summary.schemas import SummaryResult, SummaryState


def make_preprocess_node() -> Callable[[SummaryState], dict[str, Any]]:
    """Return the pure text-normalization node."""

    def preprocess(state: SummaryState) -> dict[str, Any]:
        """Collapse whitespace before the supplied text reaches the model."""
        normalized_text = re.sub(r"\s+", " ", state.get("text", "")).strip()
        if not normalized_text:
            raise AppError(ErrorCode.INVALID_PARAMETERS, "Summary text is empty")
        return {"normalized_text": normalized_text}

    return preprocess


def make_summarize_node(
    model: BaseChatModel,
) -> Callable[[SummaryState], Awaitable[dict[str, Any]]]:
    """Return the async structured-summary node bound to model."""
    structured_model = model.with_structured_output(SummaryResult)

    async def summarize(state: SummaryState) -> dict[str, Any]:
        """Request the schema-bound model response and make it state-safe."""
        try:
            prompt = SUMMARY_PROMPT.invoke(
                {
                    "text": state["normalized_text"],
                    "language": state.get("language") or "none",
                    "max_words": state.get("max_words", 200),
                }
            )
            result = await structured_model.ainvoke(prompt)
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Summary generation failed",
            ) from error
        return {"result": result.model_dump()}

    return summarize
