"""统一执行器注册表的单元测试。"""

from typing import Any

import pytest

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import (
    ExecutionContext,
    ExecutorDefinition,
)
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.schemas.tasks import SelectedMode


class _FakeExecutor:
    """返回固定结果的统一执行器替身。"""

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        return {"message": context.message}


def _definition(
    mode: SelectedMode,
    *,
    description: str = "test capability",
    is_default: bool = False,
) -> ExecutorDefinition:
    return ExecutorDefinition(
        mode=mode,
        description=description,
        executor=_FakeExecutor(),
        is_default=is_default,
    )


def _registry() -> ExecutorRegistry:
    return ExecutorRegistry(
        {"Summary": _definition(SelectedMode.WORKFLOW)},
        {
            "Solution_Planning": _definition(
                SelectedMode.DEEP_AGENT,
                is_default=True,
            )
        },
    )


def test_registry_normalizes_and_resolves_executor_names() -> None:
    registry = _registry()

    assert registry.contains(" summary ", mode=SelectedMode.WORKFLOW)
    assert registry.get("SUMMARY", mode=SelectedMode.WORKFLOW).description == "test capability"
    assert registry.names(SelectedMode.DEEP_AGENT) == ("solution_planning",)


def test_registry_rejects_duplicate_names_across_catalogs() -> None:
    with pytest.raises(ValueError, match="Duplicate executor type"):
        ExecutorRegistry(
            {"summary": _definition(SelectedMode.WORKFLOW)},
            {
                " SUMMARY ": _definition(
                    SelectedMode.DEEP_AGENT,
                    is_default=True,
                )
            },
        )


def test_registry_rejects_blank_executor_name() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        ExecutorRegistry(
            {" ": _definition(SelectedMode.WORKFLOW)},
            {
                "planner": _definition(
                    SelectedMode.DEEP_AGENT,
                    is_default=True,
                )
            },
        )


@pytest.mark.parametrize(
    "agents",
    [
        {"planner": _definition(SelectedMode.DEEP_AGENT)},
        {
            "planner": _definition(SelectedMode.DEEP_AGENT, is_default=True),
            "researcher": _definition(SelectedMode.DEEP_AGENT, is_default=True),
        },
    ],
)
def test_registry_requires_exactly_one_default_agent(agents) -> None:
    with pytest.raises(ValueError, match="exactly one default Deep Agent"):
        ExecutorRegistry(
            {"summary": _definition(SelectedMode.WORKFLOW)},
            agents,
        )


def test_registry_rejects_default_workflow() -> None:
    with pytest.raises(ValueError, match="Only a Deep Agent can be the default"):
        ExecutorRegistry(
            {"summary": _definition(SelectedMode.WORKFLOW, is_default=True)},
            {
                "planner": _definition(
                    SelectedMode.DEEP_AGENT,
                    is_default=True,
                )
            },
        )


def test_registry_exposes_default_agent_and_routing_options() -> None:
    registry = ExecutorRegistry(
        {
            "summary": _definition(
                SelectedMode.WORKFLOW,
                description="Create a structured summary",
            )
        },
        {
            "solution_planning": _definition(
                SelectedMode.DEEP_AGENT,
                description="Create implementation plans",
                is_default=True,
            )
        },
    )

    assert registry.default_agent_type == "solution_planning"
    assert [option.model_dump() for option in registry.routing_options()] == [
        {
            "executor_type": "solution_planning",
            "mode": SelectedMode.DEEP_AGENT,
            "description": "Create implementation plans",
        },
        {
            "executor_type": "summary",
            "mode": SelectedMode.WORKFLOW,
            "description": "Create a structured summary",
        },
    ]


@pytest.mark.parametrize(
    ("name", "mode", "code", "message"),
    [
        ("missing", SelectedMode.WORKFLOW, ErrorCode.INVALID_TASK_TYPE, "Unknown task type"),
        (
            "missing",
            SelectedMode.DEEP_AGENT,
            ErrorCode.INVALID_AGENT_TYPE,
            "Unknown agent type",
        ),
        (
            "summary",
            SelectedMode.DEEP_AGENT,
            ErrorCode.INVALID_AGENT_TYPE,
            "Unknown agent type",
        ),
    ],
)
def test_registry_uses_mode_specific_errors(name, mode, code, message) -> None:
    registry = _registry()

    with pytest.raises(AppError) as error:
        registry.get(name, mode=mode)

    assert error.value.code is code
    assert error.value.public_message == message
