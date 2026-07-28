from datetime import UTC

import pytest
from pydantic import ValidationError

from agent_app.schemas.events import EventSequencer, EventType, PendingEvent, TaskEvent


def test_event_type_values_match_the_sse_protocol() -> None:
    assert {event_type.value for event_type in EventType} == {
        "task.started",
        "route.selected",
        "node.started",
        "content.delta",
        "tool.started",
        "tool.completed",
        "node.completed",
        "task.completed",
        "task.failed",
    }


def test_pending_events_do_not_share_default_data() -> None:
    first = PendingEvent(type=EventType.TASK_STARTED)
    second = PendingEvent(type=EventType.TASK_STARTED)
    first.data["stage"] = "routing"

    assert second.data == {}


def test_event_sequencer_starts_at_one_and_increments() -> None:
    sequencer = EventSequencer(task_id="task-1", thread_id="thread-1")

    first = sequencer.next(EventType.TASK_STARTED, {})
    second = sequencer.next(EventType.ROUTE_SELECTED, {"selected_mode": "workflow"})

    assert [first.sequence, second.sequence] == [1, 2]
    assert first.task_id == second.task_id == "task-1"
    assert first.thread_id == second.thread_id == "thread-1"
    assert first.timestamp.tzinfo is UTC


def test_task_event_rejects_non_positive_sequences() -> None:
    with pytest.raises(ValidationError):
        TaskEvent(
            type=EventType.TASK_STARTED,
            task_id="task-1",
            thread_id="thread-1",
            sequence=0,
            timestamp=EventSequencer("task-1", "thread-1")
            .next(EventType.TASK_STARTED, {})
            .timestamp,
        )
