"""Test doubles for model-dependent workflow tests."""

from typing import Any

from agent_app.workflows.summary.schemas import SummaryResult


class FakeStructuredSummaryRunnable:
    """Async runnable returning a fixed, schema-valid summary result."""

    def __init__(self) -> None:
        self.inputs: list[Any] = []
        self.error: Exception | None = None

    async def ainvoke(self, input_value: Any) -> SummaryResult:
        """Record the prompt boundary and return the controlled model response."""
        self.inputs.append(input_value)
        if self.error is not None:
            raise self.error
        return SummaryResult(summary="测试摘要", key_points=["Alpha", "Beta"])


class FakeSummaryModel:
    """Small LangChain-compatible fake for structured summary generation."""

    def __init__(self) -> None:
        self.structured_schema: type[SummaryResult] | None = None
        self.runnable = FakeStructuredSummaryRunnable()

    def with_structured_output(self, schema: type[SummaryResult]) -> FakeStructuredSummaryRunnable:
        """Bind the requested response schema without contacting an external service."""
        self.structured_schema = schema
        return self.runnable
