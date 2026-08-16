"""DeepAgentAdapter 的单元测试。"""

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext
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


def _msg_stream(message, *, checkpoint_ns="model:uuid-1"):
    return ("messages", (message, {"langgraph_checkpoint_ns": checkpoint_ns}))


def _context(*, message="test", emit=lambda _: None, config=None):
    return ExecutionContext(
        message=message,
        messages=[HumanMessage(content=message)],
        parameters={},
        config=config or {"configurable": {"thread_id": "thread-1"}},
        emit=emit,
    )


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
    result = await adapter.run(_context(message="制定发布计划", emit=emitted.append))
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
    result = await adapter.run(_context())
    assert result == {"answer": "第一步：准备"}


@pytest.mark.asyncio
async def test_adapter_ignores_unknown_events() -> None:
    chunks = [
        _msg_stream(HumanMessage(content="skip me")),
        _msg_stream(AIMessageChunk(content="answer")),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    emitted = []
    result = await adapter.run(_context(emit=emitted.append))
    assert result == {"answer": "answer"}
    assert [e.type for e in emitted] == [EventType.CONTENT_DELTA]


@pytest.mark.asyncio
async def test_adapter_raises_when_no_answer() -> None:
    chunks = [_msg_stream(HumanMessage(content="no ai response"))]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    with pytest.raises(AppError) as error:
        await adapter.run(_context())
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
        await adapter.run(_context())
    assert error.value.code is ErrorCode.EXECUTION_FAILED
    assert error.value.public_message == "Deep Agent execution failed"


@pytest.mark.asyncio
async def test_adapter_requests_list_stream_mode() -> None:
    chunks = [_msg_stream(AIMessageChunk(content="answer"))]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    await adapter.run(_context())
    # langgraph 只在 stream_mode 为 list 时产出 (mode, payload) 元组。
    assert isinstance(adapter._runtime.stream_mode_received, list)
    assert adapter._runtime.stream_mode_received == ["messages", "updates"]


@pytest.mark.asyncio
async def test_adapter_skips_updates_mode_payloads() -> None:
    chunks = [
        ("updates", {"node": {"messages": ["irrelevant"]}}),
        _msg_stream(AIMessageChunk(content="answer")),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    emitted = []
    result = await adapter.run(_context(emit=emitted.append))
    assert result == {"answer": "answer"}
    assert [e.type for e in emitted] == [EventType.CONTENT_DELTA]


@pytest.mark.asyncio
async def test_adapter_filters_nested_subagent_chunks() -> None:
    chunks = [
        # 子代理 LLM 中间输出：最后一段节点名为 tools:（task 工具执行内的嵌套模型）。
        _msg_stream(
            AIMessageChunk(content="子代理中间输出"),
            checkpoint_ns="execute:uuid-exe|tools:uuid-tools",
        ),
        # 编排图内主代理输出：最后一段节点名为 model:（含 | 但非 tools 起始）。
        _msg_stream(
            AIMessageChunk(content="主代理回答"),
            checkpoint_ns="execute:uuid-exe|model:uuid-model",
        ),
        # task 工具的结果回传：ToolMessage 即使挂在 tools: 下也保留。
        _msg_stream(
            ToolMessage(
                content="子代理最终结果",
                name="task",
                tool_call_id="tc-2",
                status="success",
            ),
            checkpoint_ns="execute:uuid-exe|tools:uuid-tools",
        ),
    ]
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    emitted = []
    result = await adapter.run(_context(emit=emitted.append))
    assert result == {"answer": "主代理回答"}
    # 子代理内容增量被过滤；主代理 delta 与 task 的 ToolMessage 正常保留。
    assert [e.type for e in emitted] == [EventType.CONTENT_DELTA, EventType.TOOL_COMPLETED]


@pytest.mark.asyncio
async def test_adapter_keeps_main_agent_chunks_inside_orchestration_execute_node() -> None:
    """编排图 execute 节点内运行时，Deep Agent 主代理 chunk 的 ns 本就含 "|"。

    回归锁定：不得用"ns 含 |"或"深度超过基准"作为过滤判据（主代理在编排
    图内 ns 形如 execute:<uuid>|model:<uuid>，会被误杀导致 answer 恒空）；
    正确判据是最后一段节点名是否为 tools:（子代理经 task 工具调度）。
    """
    chunks = [
        _msg_stream(
            AIMessageChunk(content="主代理回答"),
            checkpoint_ns="execute:uuid-exe|model:uuid-model",
        ),
    ]
    context = _context(
        config={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "execute:uuid-exe"}}
    )
    adapter = DeepAgentAdapter(runtime=_FakeRuntime(chunks))
    result = await adapter.run(context)
    assert result == {"answer": "主代理回答"}
