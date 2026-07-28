"""Unit tests for deep agent event mapping."""

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.schemas.events import EventType


def test_ai_message_chunk_with_tool_calls_maps_to_tool_started() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_calls=[{"name": "write_todos", "args": {"todos": []}, "id": "tc-1"}],
    )
    event = map_deep_agent_event(chunk)
    assert event is not None
    assert event.type is EventType.TOOL_STARTED
    assert event.data == {"tool_name": "write_todos"}


def test_tool_message_maps_to_tool_completed_success() -> None:
    message = ToolMessage(
        content="done",
        name="write_todos",
        tool_call_id="tc-1",
        status="success",
    )
    event = map_deep_agent_event(message)
    assert event is not None
    assert event.type is EventType.TOOL_COMPLETED
    assert event.data == {"tool_name": "write_todos", "status": "success"}


def test_failed_tool_message_maps_to_tool_completed_error() -> None:
    message = ToolMessage(
        content="error: something went wrong",
        name="write_file",
        tool_call_id="tc-2",
        status="error",
    )
    event = map_deep_agent_event(message)
    assert event is not None
    assert event.type is EventType.TOOL_COMPLETED
    assert event.data == {"tool_name": "write_file", "status": "error"}


def test_ai_message_chunk_with_content_maps_to_content_delta() -> None:
    chunk = AIMessageChunk(content="这是方案的第一步")
    event = map_deep_agent_event(chunk)
    assert event is not None
    assert event.type is EventType.CONTENT_DELTA
    assert event.data == {"delta": "这是方案的第一步"}


def test_unknown_message_type_is_ignored() -> None:
    assert map_deep_agent_event(HumanMessage(content="hello")) is None


def test_plain_object_is_ignored() -> None:
    assert map_deep_agent_event(object()) is None


def test_empty_ai_chunk_returns_none() -> None:
    assert map_deep_agent_event(AIMessageChunk(content="")) is None
