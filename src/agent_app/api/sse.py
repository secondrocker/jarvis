"""SSE encoding for stable task events."""

from agent_app.schemas.events import TaskEvent


def encode_sse(event: TaskEvent) -> str:
    """Encode a TaskEvent as a raw SSE text block."""
    lines = [
        f"event: {event.type.value}",
        f"id: {event.sequence}",
        f"data: {event.model_dump_json()}",
    ]
    return "\n".join(lines) + "\n\n"
