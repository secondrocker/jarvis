"""任务请求与响应的数据传输模型。"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionMode(StrEnum):
    """任务的执行方式。"""

    AUTO = "auto"
    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class SelectedMode(StrEnum):
    """任务路由选定的执行器。"""

    WORKFLOW = "workflow"
    DEEP_AGENT = "deep_agent"


class TaskStatus(StrEnum):
    """同步 API 对外公开的任务终态。"""

    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    """任务接口接收的已校验输入。"""

    message: str
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    task_type: str | None = Field(default=None, max_length=64)
    agent_type: str | None = Field(default=None, max_length=64)
    thread_id: str | None = Field(default=None, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def strip_and_require_message(cls, value: str) -> str:
        """规范化消息，并拒绝不包含用户内容的输入。

        参数:
            value: 接口提交的原始消息文本。

        返回值:
            去除首尾空白后的消息。
        """
        message = value.strip()
        if not message:
            raise ValueError("message must not be blank")
        return message

    @model_validator(mode="after")
    def validate_execution_target(self) -> "TaskRequest":
        """确保执行模式与 workflow/agent 目标字段保持一致。

        返回值:
            校验通过的当前任务请求。
        """
        has_task_type = bool((self.task_type or "").strip())
        has_agent_type = bool((self.agent_type or "").strip())
        if has_task_type and has_agent_type:
            raise ValueError("task_type and agent_type are mutually exclusive")
        if self.execution_mode is ExecutionMode.WORKFLOW and not has_task_type:
            raise ValueError("task_type is required when execution_mode is workflow")
        if self.execution_mode is ExecutionMode.DEEP_AGENT and has_task_type:
            raise ValueError("task_type is not allowed when execution_mode is deep_agent")
        return self


class ExecutionInfo(BaseModel):
    """任务完成时返回的路由信息。"""

    selected_mode: SelectedMode
    task_type: str | None
    agent_type: str | None = None
    route_reason: str


class TaskResponse(BaseModel):
    """任务成功完成后的稳定同步响应。"""

    task_id: str
    thread_id: str
    status: Literal["completed"] = "completed"
    execution: ExecutionInfo
    result: dict[str, Any]

    @field_validator("result")
    @classmethod
    def require_non_empty_result(cls, value: dict[str, Any]) -> dict[str, Any]:
        """防止成功响应缺少调用方可见的结果。

        参数:
            value: 待写入成功响应的结果字典。

        返回值:
            校验为非空的原始结果字典。
        """
        if not value:
            raise ValueError("result must not be empty")
        return value
