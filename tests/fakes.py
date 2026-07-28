"""Test doubles for model-dependent workflow tests."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
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


class FakeTaskService:
    """In-memory service recording calls and returning a stable contract."""

    def __init__(self, *, fail_with: Any = None) -> None:
        from agent_app.schemas.events import EventType, TaskEvent
        from agent_app.schemas.tasks import (
            ExecutionInfo,
            ExecutionMode,
            SelectedMode,
            TaskRequest,
            TaskResponse,
            TaskStatus,
        )

        self._EventType = EventType
        self._TaskEvent = TaskEvent
        self._TaskResponse = TaskResponse
        self._TaskStatus = TaskStatus
        self._ExecutionInfo = ExecutionInfo
        self._SelectedMode = SelectedMode
        self._ExecutionMode = ExecutionMode
        self._TaskRequest = TaskRequest
        self.invoke_calls: list[Any] = []
        self.stream_calls: list[Any] = []
        self._fail_with = fail_with

    def preflight(self, request: Any) -> None:
        pass

    async def stream(self, request: Any) -> AsyncIterator[Any]:
        self.stream_calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        yield self._TaskEvent(
            type=self._EventType.TASK_COMPLETED,
            task_id="fake-task",
            thread_id=request.thread_id or "fake-thread",
            sequence=1,
            timestamp=datetime.now(UTC),
            data={
                "selected_mode": "workflow",
                "task_type": request.task_type,
                "route_reason": "explicit workflow",
                "result": {"summary": "fake", "key_points": []},
            },
        )

    async def invoke(self, request: Any) -> Any:
        self.invoke_calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        return self._TaskResponse(
            task_id="fake-task",
            thread_id=request.thread_id or "fake-thread",
            status=self._TaskStatus.COMPLETED,
            execution=self._ExecutionInfo(
                selected_mode=self._SelectedMode.WORKFLOW,
                task_type=request.task_type,
                route_reason="explicit workflow",
            ),
            result={"summary": "测试摘要", "key_points": ["A"]},
        )
