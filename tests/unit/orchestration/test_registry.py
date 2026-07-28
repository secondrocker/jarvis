"""WorkflowRegistry 的单元测试。"""

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.registry import Workflow, WorkflowRegistry


class _FakeWorkflow:
    """满足 Workflow 协议的最小工作流替身。"""

    async def ainvoke(self, input, config=None):
        return {"echo": input}


def test_registry_normalizes_keys() -> None:
    registry = WorkflowRegistry({"  Summary  ": _FakeWorkflow()})

    assert registry.contains("summary")
    assert registry.contains("SUMMARY")
    assert registry.contains("  summary ")


def test_registry_rejects_duplicate_normalized_keys() -> None:
    with pytest.raises(ValueError):
        WorkflowRegistry({"summary": _FakeWorkflow(), "SUMMARY": _FakeWorkflow()})


def test_registry_get_returns_workflow_for_registered_type() -> None:
    workflow = _FakeWorkflow()
    registry = WorkflowRegistry({"summary": workflow})

    assert registry.get("summary") is workflow


def test_registry_get_raises_for_missing_task_type() -> None:
    registry = WorkflowRegistry({"summary": _FakeWorkflow()})

    with pytest.raises(AppError) as error:
        registry.get("translate")

    assert error.value.code is ErrorCode.INVALID_TASK_TYPE
    assert error.value.public_message == "Unknown task type"


def test_registry_names_returns_sorted_normalized_keys() -> None:
    registry = WorkflowRegistry({"summary": _FakeWorkflow(), "Extract": _FakeWorkflow()})

    assert registry.names() == ("extract", "summary")


def test_workflow_protocol_is_satisfied_by_fake() -> None:
    workflow: Workflow = _FakeWorkflow()
    assert hasattr(workflow, "ainvoke")
