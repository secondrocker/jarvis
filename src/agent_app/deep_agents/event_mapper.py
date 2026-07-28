"""Map deepagents stream chunks to project-normalized PendingEvents."""

from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from agent_app.schemas.events import EventType, PendingEvent


def map_deep_agent_event(message: Any) -> PendingEvent | None:
    """Translate a stream message to a PendingEvent, or None to ignore."""
    if isinstance(message, AIMessageChunk):
        if message.tool_calls:
            tool_name = message.tool_calls[0].get("name", "unknown")
            return PendingEvent(
                type=EventType.TOOL_STARTED,
                data={"tool_name": tool_name},
            )
        if message.content:
            return PendingEvent(
                type=EventType.CONTENT_DELTA,
                data={"delta": message.content},
            )
        return None

    if isinstance(message, ToolMessage):
        status = getattr(message, "status", "success")
        return PendingEvent(
            type=EventType.TOOL_COMPLETED,
            data={
                "tool_name": getattr(message, "name", "unknown"),
                "status": status,
            },
        )

    return None
