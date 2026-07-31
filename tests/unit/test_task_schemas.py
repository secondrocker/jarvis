import pytest
from pydantic import ValidationError

from agent_app.schemas.tasks import (
    ExecutionInfo,
    ExecutionMode,
    SelectedMode,
    TaskRequest,
    TaskResponse,
)


def test_task_request_defaults_to_auto_and_empty_parameters() -> None:
    request = TaskRequest(message="制定一个发布方案")

    assert request.execution_mode is ExecutionMode.AUTO
    assert request.task_type is None
    assert request.agent_type is None
    assert request.thread_id is None
    assert request.parameters == {}


def test_task_request_strips_message_before_exposing_it() -> None:
    request = TaskRequest(message="  制定一个发布方案  ")

    assert request.message == "制定一个发布方案"


def test_task_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(message="   ")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("thread_id", "x" * 129),
        ("task_type", "x" * 65),
        ("agent_type", "x" * 65),
    ],
)
def test_task_request_rejects_identifiers_over_the_public_limit(
    field_name: str, value: str
) -> None:
    with pytest.raises(ValidationError):
        TaskRequest(message="总结", **{field_name: value})


def test_explicit_workflow_requires_non_blank_task_type() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(message="总结", execution_mode=ExecutionMode.WORKFLOW, task_type=" ")


def test_task_request_rejects_both_target_types() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(message="执行", task_type="summary", agent_type="solution_planning")


def test_explicit_workflow_rejects_agent_type() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            message="执行",
            execution_mode=ExecutionMode.WORKFLOW,
            agent_type="solution_planning",
        )


def test_explicit_deep_agent_rejects_task_type() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            message="执行",
            execution_mode=ExecutionMode.DEEP_AGENT,
            task_type="summary",
        )


def test_task_response_has_the_stable_completed_shape() -> None:
    response = TaskResponse(
        task_id="task-1",
        thread_id="thread-1",
        execution=ExecutionInfo(
            selected_mode=SelectedMode.WORKFLOW,
            task_type="summary",
            agent_type=None,
            route_reason="registered task type",
        ),
        result={"summary": "摘要"},
    )

    assert response.status == "completed"
    assert response.result == {"summary": "摘要"}


def test_task_response_rejects_an_empty_result() -> None:
    execution = ExecutionInfo(
        selected_mode=SelectedMode.DEEP_AGENT,
        task_type=None,
        agent_type="solution_planning",
        route_reason="open-ended planning task",
    )

    with pytest.raises(ValidationError):
        TaskResponse(
            task_id="task-1",
            thread_id="thread-1",
            execution=execution,
            result={},
        )


def test_execution_info_exposes_selected_agent_type() -> None:
    execution = ExecutionInfo(
        selected_mode=SelectedMode.DEEP_AGENT,
        task_type=None,
        agent_type="solution_planning",
        route_reason="explicit deep agent",
    )

    assert execution.agent_type == "solution_planning"
