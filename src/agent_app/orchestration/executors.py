"""顶层编排可调用的统一执行器契约。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agent_app.schemas.events import PendingEvent
from agent_app.schemas.tasks import SelectedMode

EmitFn = Callable[[PendingEvent], None]


@dataclass(frozen=True)
class ExecutionContext:
    """一次执行所需的输入、会话状态和事件出口。"""

    message: str
    messages: list[Any]
    parameters: dict[str, Any]
    config: RunnableConfig
    emit: EmitFn


@runtime_checkable
class Executor(Protocol):
    """workflow 与 Deep Agent 共同实现的执行契约。"""

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        """执行一次任务并返回调用方可见的结果。"""
        ...


@dataclass(frozen=True)
class ExecutorDefinition:
    """执行器实例及其路由元数据。"""

    mode: SelectedMode
    description: str
    executor: Executor
    is_default: bool = False


class RoutingOption(BaseModel):
    """提供给路由模型的稳定执行器能力描述。"""

    executor_type: str
    mode: SelectedMode
    description: str
