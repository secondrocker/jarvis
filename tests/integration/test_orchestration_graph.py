"""Integration tests for the top-level orchestration graph."""

import pytest

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.schemas.tasks import SelectedMode


class _FakeWorkflow:
    """Summary workflow fake returning a fixed structured result."""

    async def ainvoke(self, input, config=None):
        return {"summary": "测试摘要", "key_points": ["A", "B"]}


class _FakeRouterModel:
    """Fake router model that returns a deep-agent decision for non-summary text."""

    def __init__(self, decision):
        self._decision = decision
        self._schema = None

    def with_structured_output(self, schema):
        self._schema = schema

        class _R:
            def __init__(self, decision):
                self._d = decision

            async def ainvoke(self, prompt):
                return self._d

        return _R(self._decision)


class _FakeDeepAgentRuntime:
    """Fake runtime yielding one content delta."""

    async def astream(self, input, config, *, stream_mode):
        from langchain_core.messages import AIMessageChunk
        yield ("messages", (AIMessageChunk(content="这是方案"), {}))


def _registry():
    return WorkflowRegistry({"summary": _FakeWorkflow()})


@pytest.mark.asyncio
async def test_graph_dispatches_workflow_and_preserves_route_metadata() -> None:
    router = TaskRouter(registry=_registry(), model=_FakeRouterModel(None))
    deep_agent = DeepAgentAdapter(runtime=_FakeDeepAgentRuntime())
    graph = build_orchestration_graph(
        router=router, registry=_registry(), deep_agent=deep_agent,
        checkpointer=create_checkpointer(),
    )
    output = await graph.ainvoke(
        {"task_id": "task-1", "thread_id": "thread-1", "message": "总结文本",
         "execution_mode": "auto", "requested_task_type": None, "parameters": {}},
        {"configurable": {"thread_id": "thread-1"}},
    )
    assert output["selected_mode"] == "workflow"
    assert output["selected_task_type"] == "summary"
    assert output["result"]["summary"] == "测试摘要"


@pytest.mark.asyncio
async def test_graph_dispatches_deep_agent_for_open_ended_task() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision
    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT, task_type=None,
            is_ambiguous=False, reason="open-ended planning",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    deep_agent = DeepAgentAdapter(runtime=_FakeDeepAgentRuntime())
    graph = build_orchestration_graph(
        router=router, registry=_registry(), deep_agent=deep_agent,
        checkpointer=create_checkpointer(),
    )
    output = await graph.ainvoke(
        {"task_id": "task-2", "thread_id": "thread-2", "message": "制定发布计划",
         "execution_mode": "auto", "requested_task_type": None, "parameters": {}},
        {"configurable": {"thread_id": "thread-2"}},
    )
    assert output["selected_mode"] == "deep_agent"
    assert output["result"]["answer"] == "这是方案"
