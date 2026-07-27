"""Workflow registry for explicit task-type registration."""

from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

from agent_app.errors import AppError, ErrorCode


@runtime_checkable
class Workflow(Protocol):
    """Contract for a registered fixed workflow."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow and return a result dictionary."""
        ...


def _normalize(task_type: str) -> str:
    """Normalize a task type for case-insensitive lookup."""
    return task_type.strip().lower()


class WorkflowRegistry:
    """Holds explicitly registered fixed workflows keyed by normalized task type."""

    def __init__(self, workflows: dict[str, Workflow]) -> None:
        """Validate and store registrations with normalized keys."""
        self._workflows: dict[str, Workflow] = {}
        for raw_name, workflow in workflows.items():
            key = _normalize(raw_name)
            if key in self._workflows:
                raise ValueError(f"Duplicate task type after normalization: {key}")
            self._workflows[key] = workflow

    def contains(self, task_type: str) -> bool:
        """Return whether a normalized task type is registered."""
        return _normalize(task_type) in self._workflows

    def get(self, task_type: str) -> Workflow:
        """Return a registered workflow or raise INVALID_TASK_TYPE."""
        key = _normalize(task_type)
        if key not in self._workflows:
            raise AppError(ErrorCode.INVALID_TASK_TYPE, "Unknown task type")
        return self._workflows[key]

    def names(self) -> tuple[str, ...]:
        """Return sorted normalized task type names."""
        return tuple(sorted(self._workflows))
