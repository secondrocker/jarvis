"""Tests proving thread isolation and multi-turn checkpoint behavior."""

import pytest

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.router import TaskRouter


class _FakeWorkflow:
    async def ainvoke(self, input, config=None):
        return {"summary": "ok", "key_points": []}


class _FakeRouterModel:
    def with_structured_output(self, schema):
        class _R:
            async def ainvoke(self, prompt):
                pass

        return _R()


def _graph():
    router = TaskRouter(
        registry=WorkflowRegistry({"summary": _FakeWorkflow()}),
        model=_FakeRouterModel(),
    )
    return build_orchestration_graph(
        router=router,
        registry=WorkflowRegistry({"summary": _FakeWorkflow()}),
        deep_agent=DeepAgentAdapter(runtime=None),
        checkpointer=create_checkpointer(),
    )


@pytest.mark.asyncio
async def test_same_thread_accumulates_messages_across_turns() -> None:
    graph = _graph()
    config = {"configurable": {"thread_id": "shared"}}
    await graph.ainvoke(
        {
            "task_id": "t1",
            "thread_id": "shared",
            "message": "总结第一段",
            "execution_mode": "auto",
            "requested_task_type": None,
            "parameters": {},
        },
        config,
    )
    state_after_first = await graph.aget_state(config)
    await graph.ainvoke(
        {
            "task_id": "t2",
            "thread_id": "shared",
            "message": "总结第二段",
            "execution_mode": "auto",
            "requested_task_type": None,
            "parameters": {},
        },
        config,
    )
    state_after_second = await graph.aget_state(config)
    messages_first = state_after_first.values.get("messages", [])
    messages_second = state_after_second.values.get("messages", [])
    assert len(messages_first) == 1
    assert len(messages_second) == 2


@pytest.mark.asyncio
async def test_different_threads_are_isolated() -> None:
    graph = _graph()
    await graph.ainvoke(
        {
            "task_id": "t1",
            "thread_id": "thread-a",
            "message": "总结",
            "execution_mode": "auto",
            "requested_task_type": None,
            "parameters": {},
        },
        {"configurable": {"thread_id": "thread-a"}},
    )
    await graph.ainvoke(
        {
            "task_id": "t2",
            "thread_id": "thread-b",
            "message": "总结",
            "execution_mode": "auto",
            "requested_task_type": None,
            "parameters": {},
        },
        {"configurable": {"thread_id": "thread-b"}},
    )
    state_a = await graph.aget_state({"configurable": {"thread_id": "thread-a"}})
    state_b = await graph.aget_state({"configurable": {"thread_id": "thread-b"}})
    assert len(state_a.values.get("messages", [])) == 1
    assert len(state_b.values.get("messages", [])) == 1
