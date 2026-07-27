"""Task request and response data transfer objects."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionMode(StrEnum):
    """How the task should be executed."""

    AUTO = "auto"
    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class SelectedMode(StrEnum):
    """The executor selected by task routing."""

    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class TaskStatus(StrEnum):
    """Terminal task statuses exposed by the synchronous API."""

    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    """Validated input accepted by task endpoints."""

    message: str
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    task_type: str | None = Field(default=None, max_length=64)
    thread_id: str | None = Field(default=None, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def strip_and_require_message(cls, value: str) -> str:
        """Normalize messages while rejecting input with no user content."""
        message = value.strip()
        if not message:
            raise ValueError("message must not be blank")
        return message

    @model_validator(mode="after")
    def require_task_type_for_explicit_workflow(self) -> "TaskRequest":
        """Require a workflow name before the request reaches service preflight."""
        if self.execution_mode is ExecutionMode.WORKFLOW and not (self.task_type or "").strip():
            raise ValueError("task_type is required when execution_mode is workflow")
        return self


class ExecutionInfo(BaseModel):
    """Routing information returned with a completed task."""

    selected_mode: SelectedMode
    task_type: str | None
    route_reason: str


class TaskResponse(BaseModel):
    """Stable synchronous response for a successfully completed task."""

    task_id: str
    thread_id: str
    status: Literal["completed"] = "completed"
    execution: ExecutionInfo
    result: dict[str, Any]

    @field_validator("result")
    @classmethod
    def require_non_empty_result(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Prevent successful responses without a consumer-visible result."""
        if not value:
            raise ValueError("result must not be empty")
        return value
