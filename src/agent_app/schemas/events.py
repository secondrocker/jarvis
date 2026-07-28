"""同步与流式任务执行共用的标准化内部事件。"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """稳定 SSE 协议中的事件名称。"""

    TASK_STARTED = "task.started"
    ROUTE_SELECTED = "route.selected"
    NODE_STARTED = "node.started"
    CONTENT_DELTA = "content.delta"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    NODE_COMPLETED = "node.completed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"


class PendingEvent(BaseModel):
    """工作流在附加任务元数据前发出的待处理事件。"""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class TaskEvent(BaseModel):
    """已完成排序、可供同步聚合或 SSE 编码的任务事件。"""

    type: EventType
    task_id: str
    thread_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class EventSequencer:
    """为事件附加稳定的任务元数据和单调递增序号。"""

    def __init__(self, task_id: str, thread_id: str) -> None:
        """初始化任务事件序号生成器。

        参数:
            task_id: 当前执行的唯一任务标识。
            thread_id: 用于关联多轮会话的线程标识。
        """
        self._task_id = task_id
        self._thread_id = thread_id
        self._sequence = 0

    def next(self, event_type: EventType, data: dict[str, Any]) -> TaskEvent:
        """为当前任务的唯一有序事件流创建下一个事件。

        参数:
            event_type: 稳定协议中的事件类型。
            data: 事件携带的业务数据。

        返回值:
            已附加任务标识、线程标识和递增序号的任务事件。
        """
        self._sequence += 1
        return TaskEvent(
            type=event_type,
            task_id=self._task_id,
            thread_id=self._thread_id,
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            data=data,
        )
