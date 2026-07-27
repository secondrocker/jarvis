"""Normalized internal events used by synchronous and streaming task execution."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Event names in the stable SSE protocol."""

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
    """An event emitted by a workflow before task metadata is attached."""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class TaskEvent(BaseModel):
    """A fully sequenced event ready for synchronous aggregation or SSE encoding."""

    type: EventType
    task_id: str
    thread_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class EventSequencer:
    """Attach stable task metadata and monotonically increasing sequence numbers."""

    def __init__(self, task_id: str, thread_id: str) -> None:
        self._task_id = task_id
        self._thread_id = thread_id
        self._sequence = 0

    def next(self, event_type: EventType, data: dict[str, Any]) -> TaskEvent:
        """Create the next event for this task's single ordered event stream."""
        self._sequence += 1
        return TaskEvent(
            type=event_type,
            task_id=self._task_id,
            thread_id=self._thread_id,
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            data=data,
        )
