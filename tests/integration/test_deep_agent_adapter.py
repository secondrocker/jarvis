"""DeepAgentAdapter 与真实 deepagents 运行时的集成测试。

用 ScriptedModel 驱动真实 ``create_deep_agent``（含子代理），验证流式
路径端到端可用——FakeRuntime 替身无法暴露 langgraph 流模式形状问题。
"""

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from fakes import ScriptedModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.events import EventType


def _restricted_runtime(model: ScriptedModel, *, subagents=None) -> object:
    """按工厂同等约束构建真实受限运行时。"""
    return create_deep_agent(
        model=model,
        tools=None,
        subagents=list(subagents) if subagents else None,
        skills=None,
        backend=StateBackend(),
        checkpointer=MemorySaver(),
    )


@pytest.mark.asyncio
async def test_adapter_streams_real_runtime_without_subagents() -> None:
    model = ScriptedModel.from_scripts(["这是一个发布方案"])
    runtime = _restricted_runtime(model)
    adapter = DeepAgentAdapter(runtime=runtime)
    emitted = []
    context = ExecutionContext(
        message="制定发布方案",
        messages=[HumanMessage(content="制定发布方案")],
        parameters={},
        config={"configurable": {"thread_id": "thread-a"}},
        emit=emitted.append,
    )
    result = await adapter.run(context)
    assert result == {"answer": "这是一个发布方案"}
    assert EventType.CONTENT_DELTA in [e.type for e in emitted]


@pytest.mark.asyncio
async def test_adapter_streams_real_runtime_with_subagent() -> None:
    subagent = {
        "name": "researcher",
        "description": "查询数据并返回结果",
        "system_prompt": "你是数据查询专员。",
        "tools": [],
    }
    # 脚本序列：主代理调 task → 子代理输出文本 → 主代理写最终报告。
    model = ScriptedModel.from_scripts(
        [
            {
                "tool_call": {
                    "name": "task",
                    "args": {
                        "description": "查询钢材价格",
                        "subagent_type": "researcher",
                    },
                    "id": "tc-task",
                }
            },
            "广东 2026-07 螺纹钢 HRB400 信息价 3850 元/吨",
            "最终报告：钢材价格已获取。",
        ]
    )
    runtime = _restricted_runtime(model, subagents=[subagent])
    adapter = DeepAgentAdapter(runtime=runtime)
    emitted = []
    context = ExecutionContext(
        message="查询广东钢材价格",
        messages=[HumanMessage(content="查询广东钢材价格")],
        parameters={},
        config={"configurable": {"thread_id": "thread-b"}},
        emit=emitted.append,
    )
    result = await adapter.run(context)
    assert result == {"answer": "最终报告：钢材价格已获取。"}
    types = [e.type for e in emitted]
    # task 工具的启停事件保留；子代理中间文本不进入 answer。
    assert EventType.TOOL_STARTED in types
    assert EventType.TOOL_COMPLETED in types
    deltas = [e.data["delta"] for e in emitted if e.type is EventType.CONTENT_DELTA]
    assert "".join(deltas) == "最终报告：钢材价格已获取。"
