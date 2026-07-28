"""DeepAgentAdapter 的单元测试。"""

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.errors import AppError, ErrorCode
from agent_app.schemas.events import EventType


class _FakeRuntime:
    """产生受控数据块流的 DeepAgentRuntime 替身。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.input_received = None
        self.config_received = None
        self.stream_mode_received = None

    async def astream(self, input, config, *, stream_mode):
        self.input_received = input
        self.config_received = config
        self.stream_mode_received = stream_mode
        for chunk in self._chunks:
            yield chunk


def _msg_stream(message):
    return ("messages", (message, {}))


@pytest.mark.asyncio
async def test_adapter_maps_events_and_returns_answer() -> None:
    chunks = [
        _msg_stream(
            AIMessageChunk(
                content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "tc-1"}]
            )
        ),
        _msg_stream(
            ToolMessage(
                content="ok",
                name="write_todos",
                tool_call_id="tc-1",
                status="success",
            )
        ),
        _msg_stream(AIMessageChunk(content="发布方案")),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    emitted = []
    result = await adapter.run(
        message="制定发布计划",
        messages=[],
        config={"configurable": {"thread_id": "thread-1"}},
        emit=emitted.append,
    )
    assert result == {"answer": "发布方案"}
    assert [e.type for e in emitted] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.CONTENT_DELTA,
    ]
    assert emitted[2].data == {"delta": "发布方案"}


@pytest.mark.asyncio
async def test_adapter_accumulates_multi_chunk_answer() -> None:
    chunks = [
        _msg_stream(AIMessageChunk(content="第一步")),
        _msg_stream(AIMessageChunk(content="：")),
        _msg_stream(AIMessageChunk(content="准备")),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    result = await adapter.run(
        message="test",
        messages=[],
        config={},
        emit=lambda _: None,
    )
    assert result == {"answer": "第一步：准备"}


@pytest.mark.asyncio
async def test_adapter_ignores_unknown_events() -> None:
    chunks = [
        _msg_stream(HumanMessage(content="skip me")),
        _msg_stream(AIMessageChunk(content="answer")),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    emitted = []
    result = await adapter.run(
        message="test",
        messages=[],
        config={},
        emit=emitted.append,
    )
    assert result == {"answer": "answer"}
    assert [e.type for e in emitted] == [EventType.CONTENT_DELTA]


@pytest.mark.asyncio
async def test_adapter_raises_when_no_answer() -> None:
    chunks = [_msg_stream(HumanMessage(content="no ai response"))]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    with pytest.raises(AppError) as error:
        await adapter.run(
            message="test",
            messages=[],
            config={},
            emit=lambda _: None,
        )
    assert error.value.code is ErrorCode.EXECUTION_FAILED
    assert error.value.public_message == "Deep Agent returned no answer"


@pytest.mark.asyncio
async def test_adapter_maps_runtime_exception_to_execution_failed() -> None:
    class _ErroringRuntime:
        async def astream(self, input, config, *, stream_mode):
            raise RuntimeError("internal failure")
            yield  # pragma: no cover — keeps this an async generator

    adapter = DeepAgentAdapter(runtime=_ErroringRuntime())
    with pytest.raises(AppError) as error:
        await adapter.run(
            message="test",
            messages=[],
            config={},
            emit=lambda _: None,
        )
    assert error.value.code is ErrorCode.EXECUTION_FAILED
    assert error.value.public_message == "Deep Agent execution failed"
