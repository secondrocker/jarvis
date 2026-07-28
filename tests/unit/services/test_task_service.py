"""Unit tests for TaskService stream ordering and invoke aggregation."""

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.schemas.events import EventType
from agent_app.schemas.tasks import ExecutionMode, TaskRequest
from agent_app.services.task_service import TaskService


class _FakeSuccessGraph:
    """Fake graph yielding custom events and a final values snapshot."""

    async def astream(self, input, config, *, stream_mode):
        yield ("custom", {"pending_event": {
            "type": "route.selected",
            "data": {"selected_mode": "workflow", "task_type": "summary"},
        }})
        yield ("custom", {"pending_event": {
            "type": "node.started", "data": {"node": "workflow.summary"},
        }})
        yield ("custom", {"pending_event": {
            "type": "node.completed", "data": {"node": "workflow.summary"},
        }})
        yield ("values", {
            "selected_mode": "workflow",
            "selected_task_type": "summary",
            "route_reason": "Summary intent detected",
            "result": {"summary": "测试摘要", "key_points": ["A"]},
        })


@pytest.fixture
def fake_success_graph():
    return _FakeSuccessGraph()


@pytest.mark.asyncio
async def test_stream_wraps_graph_events_with_monotonic_sequence(fake_success_graph) -> None:
    service = TaskService(
        graph=fake_success_graph,
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(TaskRequest(message="总结"))]
    assert [event.type for event in events] == [
        EventType.TASK_STARTED,
        EventType.ROUTE_SELECTED,
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
        EventType.TASK_COMPLETED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_invoke_consumes_the_same_stream(fake_success_graph) -> None:
    service = TaskService(
        graph=fake_success_graph,
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    response = await service.invoke(TaskRequest(message="总结"))
    assert response.status == "completed"
    assert response.result == {"summary": "测试摘要", "key_points": ["A"]}
    assert response.execution.selected_mode.value == "workflow"
    assert response.execution.task_type == "summary"


@pytest.mark.asyncio
async def test_stream_generates_thread_id_when_missing(fake_success_graph) -> None:
    service = TaskService(
        graph=fake_success_graph,
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(TaskRequest(message="总结"))]
    assert all(e.thread_id for e in events)
    assert all(e.thread_id == events[0].thread_id for e in events)


@pytest.mark.asyncio
async def test_stream_preserves_caller_thread_id(fake_success_graph) -> None:
    service = TaskService(
        graph=fake_success_graph,
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(
        TaskRequest(message="总结", thread_id="my-thread")
    )]
    assert all(e.thread_id == "my-thread" for e in events)


def test_preflight_rejects_unregistered_workflow_task_type() -> None:
    service = TaskService(
        graph=_FakeSuccessGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    with pytest.raises(AppError) as error:
        service.preflight(
            TaskRequest(message="x", execution_mode=ExecutionMode.WORKFLOW, task_type="translate")
        )
    assert error.value.code is ErrorCode.INVALID_TASK_TYPE


def test_preflight_allows_auto_without_task_type() -> None:
    service = TaskService(
        graph=_FakeSuccessGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    service.preflight(TaskRequest(message="x"))


def test_preflight_allows_deep_agent_without_registered_task_type() -> None:
    service = TaskService(
        graph=_FakeSuccessGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    service.preflight(
        TaskRequest(message="x", execution_mode=ExecutionMode.DEEP_AGENT, task_type="unknown")
    )


@pytest.mark.asyncio
async def test_stream_maps_error_state_to_task_failed() -> None:
    class _ErrorGraph:
        async def astream(self, input, config, *, stream_mode):
            yield ("values", {"error": {"code": "EXECUTION_FAILED", "message": "boom"}})

    service = TaskService(
        graph=_ErrorGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(TaskRequest(message="x"))]
    assert events[-1].type is EventType.TASK_FAILED
    assert events[-1].data["reason"] == "boom"


@pytest.mark.asyncio
async def test_stream_sanitizes_unknown_exceptions() -> None:
    class _CrashGraph:
        async def astream(self, input, config, *, stream_mode):
            raise RuntimeError("secret stack trace")
            yield  # pragma: no cover — keeps this an async generator

    service = TaskService(
        graph=_CrashGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(TaskRequest(message="x"))]
    assert events[-1].type is EventType.TASK_FAILED
    assert events[-1].data["code"] == "INTERNAL_ERROR"
    assert "secret" not in events[-1].data["reason"]


@pytest.mark.asyncio
async def test_invoke_raises_app_error_on_failed_task() -> None:
    class _ErrorGraph:
        async def astream(self, input, config, *, stream_mode):
            yield ("values", {"error": {"code": "EXECUTION_FAILED", "message": "boom"}})

    service = TaskService(
        graph=_ErrorGraph(),
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    with pytest.raises(AppError) as error:
        await service.invoke(TaskRequest(message="x"))
    assert error.value.code is ErrorCode.EXECUTION_FAILED
    assert error.value.public_message == "boom"


@pytest.mark.asyncio
async def test_stream_emits_exactly_one_terminal_event(fake_success_graph) -> None:
    service = TaskService(
        graph=fake_success_graph,
        registered_task_types={"summary"},
        task_timeout_seconds=5,
    )
    events = [event async for event in service.stream(TaskRequest(message="总结"))]
    terminal = [e for e in events if e.type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED)]
    assert len(terminal) == 1
    assert terminal[0].type is EventType.TASK_COMPLETED
