"""统一执行器顶层编排图的集成测试。"""

from typing import Any

import pytest

from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.orchestration.executors import ExecutionContext, ExecutorDefinition
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.schemas.tasks import SelectedMode


class _FakeExecutor:
    """记录上下文并返回固定结果或抛出受控异常。"""

    def __init__(self, result: dict[str, Any], error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.contexts: list[ExecutionContext] = []

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeRouterModel:
    def __init__(self, decision=None):
        self.decision = decision

    def with_structured_output(self, schema):
        decision = self.decision

        class _Runnable:
            async def ainvoke(self, prompt):
                return decision

        return _Runnable()


def _definition(
    mode: SelectedMode,
    executor: _FakeExecutor,
    *,
    is_default: bool = False,
) -> ExecutorDefinition:
    return ExecutorDefinition(
        mode=mode,
        description=f"{mode.value} test executor",
        executor=executor,
        is_default=is_default,
    )


def _build_graph(*, failing_research: bool = False):
    summary = _FakeExecutor({"summary": "测试摘要"})
    planning = _FakeExecutor({"answer": "规划方案"})
    research = _FakeExecutor(
        {"answer": "研究结论"},
        RuntimeError("secret executor details") if failing_research else None,
    )
    registry = ExecutorRegistry(
        {"summary": _definition(SelectedMode.WORKFLOW, summary)},
        {
            "solution_planning": _definition(
                SelectedMode.DEEP_AGENT,
                planning,
                is_default=True,
            ),
            "research": _definition(SelectedMode.DEEP_AGENT, research),
        },
    )
    router = TaskRouter(registry=registry, model=_FakeRouterModel())
    graph = build_orchestration_graph(
        router=router,
        registry=registry,
        checkpointer=create_checkpointer(),
    )
    return graph, summary, planning, research


def _input(**overrides):
    graph_input = {
        "task_id": "task-1",
        "thread_id": "thread-1",
        "message": "执行任务",
        "execution_mode": "auto",
        "requested_task_type": None,
        "requested_agent_type": None,
        "parameters": {},
    }
    graph_input.update(overrides)
    return graph_input


@pytest.mark.asyncio
async def test_graph_executes_named_workflow_through_unified_node() -> None:
    graph, summary, planning, research = _build_graph()

    output = await graph.ainvoke(
        _input(requested_task_type="summary"),
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert output["selected_mode"] == "workflow"
    assert output["selected_executor_type"] == "summary"
    assert output["result"] == {"summary": "测试摘要"}
    assert len(summary.contexts) == 1
    assert planning.contexts == []
    assert research.contexts == []


@pytest.mark.asyncio
async def test_graph_dispatches_different_named_agent_without_graph_changes() -> None:
    graph, summary, planning, research = _build_graph()

    output = await graph.ainvoke(
        _input(
            message="调查证据",
            execution_mode="deep_agent",
            requested_agent_type="research",
        ),
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert output["selected_mode"] == "deep_agent"
    assert output["selected_executor_type"] == "research"
    assert output["result"] == {"answer": "研究结论"}
    assert summary.contexts == []
    assert planning.contexts == []
    assert len(research.contexts) == 1
    assert [message.content for message in research.contexts[0].messages] == ["调查证据"]


@pytest.mark.asyncio
async def test_graph_sanitizes_unexpected_executor_errors() -> None:
    graph, _, _, _ = _build_graph(failing_research=True)

    output = await graph.ainvoke(
        _input(execution_mode="deep_agent", requested_agent_type="research"),
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert output["error"] == {
        "code": "EXECUTION_FAILED",
        "message": "Executor execution failed",
    }
    assert "secret" not in str(output["error"])


@pytest.mark.asyncio
async def test_successful_follow_up_clears_previous_error_state() -> None:
    graph, _, _, _ = _build_graph(failing_research=True)
    config = {"configurable": {"thread_id": "shared-thread"}}

    failed = await graph.ainvoke(
        _input(
            thread_id="shared-thread",
            execution_mode="deep_agent",
            requested_agent_type="research",
        ),
        config,
    )
    succeeded = await graph.ainvoke(
        _input(
            thread_id="shared-thread",
            requested_task_type="summary",
        ),
        config,
    )

    assert failed["error"]["code"] == "EXECUTION_FAILED"
    assert succeeded["error"] is None
    assert succeeded["result"] == {"summary": "测试摘要"}


@pytest.mark.asyncio
async def test_agent_follow_up_receives_previous_user_and_assistant_messages() -> None:
    graph, _, planning, _ = _build_graph()
    config = {"configurable": {"thread_id": "agent-thread"}}

    await graph.ainvoke(
        _input(
            thread_id="agent-thread",
            message="第一轮问题",
            execution_mode="deep_agent",
        ),
        config,
    )
    await graph.ainvoke(
        _input(
            thread_id="agent-thread",
            message="第二轮问题",
            execution_mode="deep_agent",
        ),
        config,
    )

    assert [message.content for message in planning.contexts[1].messages] == [
        "第一轮问题",
        "规划方案",
        "第二轮问题",
    ]
