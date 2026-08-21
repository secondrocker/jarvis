"""Deep Agent 事件映射的单元测试。"""

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.schemas.events import EventType


def test_ai_message_chunk_with_tool_calls_maps_to_tool_started() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_calls=[{"name": "write_todos", "args": {"todos": []}, "id": "tc-1"}],
    )
    events = map_deep_agent_event(chunk)
    assert events is not None
    assert events[0].type is EventType.TOOL_STARTED
    assert events[0].data == {"tool_name": "write_todos"}


def test_empty_tool_name_in_chunk_is_filtered() -> None:
    """流式工具调用的中间分片 name 为空时不应产生空 tool_name 事件。"""
    chunk = AIMessageChunk(
        content="",
        tool_calls=[{"name": "", "args": {}, "id": "tc-1"}],
    )
    assert map_deep_agent_event(chunk) is None


def test_multiple_tool_calls_in_one_chunk_map_to_separate_events() -> None:
    """同一 chunk 中多个非空工具调用应分别产生事件。"""
    chunk = AIMessageChunk(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {}, "id": "tc-1"},
            {"name": "write_todos", "args": {}, "id": "tc-2"},
        ],
    )
    events = map_deep_agent_event(chunk)
    assert events is not None
    assert [e.data["tool_name"] for e in events] == ["read_file", "write_todos"]


def test_tool_message_maps_to_tool_completed_success() -> None:
    message = ToolMessage(
        content="done",
        name="write_todos",
        tool_call_id="tc-1",
        status="success",
    )
    events = map_deep_agent_event(message)
    assert events is not None
    assert events[0].type is EventType.TOOL_COMPLETED
    assert events[0].data == {"tool_name": "write_todos", "status": "success"}


def test_failed_tool_message_maps_to_tool_completed_error() -> None:
    message = ToolMessage(
        content="error: something went wrong",
        name="write_file",
        tool_call_id="tc-2",
        status="error",
    )
    events = map_deep_agent_event(message)
    assert events is not None
    assert events[0].type is EventType.TOOL_COMPLETED
    assert events[0].data == {"tool_name": "write_file", "status": "error"}


def test_ai_message_chunk_with_content_maps_to_content_delta() -> None:
    chunk = AIMessageChunk(content="这是方案的第一步")
    events = map_deep_agent_event(chunk)
    assert events is not None
    assert events[0].type is EventType.CONTENT_DELTA
    assert events[0].data == {"delta": "这是方案的第一步"}


def test_unknown_message_type_is_ignored() -> None:
    assert map_deep_agent_event(HumanMessage(content="hello")) is None


def test_plain_object_is_ignored() -> None:
    assert map_deep_agent_event(object()) is None


def test_empty_ai_chunk_returns_none() -> None:
    assert map_deep_agent_event(AIMessageChunk(content="")) is None
