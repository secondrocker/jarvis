"""使用完整装配图和测试替身的端到端验收测试。"""

from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
import pytest
from fakes import FakeObjectStorage

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.infrastructure.checkpoint import create_checkpointer
from agent_app.orchestration.executors import ExecutorDefinition
from agent_app.orchestration.graph import build_orchestration_graph
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.schemas.events import EventType
from agent_app.schemas.tasks import (
    ExecutionMode,
    SelectedMode,
    TaskRequest,
)
from agent_app.services.task_service import TaskService
from agent_app.workflows import create_workflows


class _FakeSummaryModel:
    """通过 with_structured_output 返回摘要的模型替身。"""

    def with_structured_output(self, schema):
        class _R:
            async def ainvoke(self, prompt):
                return schema(summary="这是摘要", key_points=["A", "B"])

        return _R()


class _FakeRouterModel:
    """为非摘要文本返回 Deep Agent 决策的模型替身。"""

    def with_structured_output(self, schema):
        class _R:
            async def ainvoke(self, prompt):
                return schema(
                    selected_mode=SelectedMode.DEEP_AGENT,
                    executor_type="solution_planning",
                    is_ambiguous=False,
                    reason="open-ended planning",
                )

        return _R()


class _FakeDeepAgentRuntime:
    """只产生一个答案块的 Deep Agent 运行时替身。"""

    async def astream(self, input, config, *, stream_mode, **kwargs):
        from langchain_core.messages import AIMessageChunk

        yield ("messages", (AIMessageChunk(content="这是一个发布方案"), {}))


def _build_service(checkpointer=None) -> TaskService:
    """使用无需网络的替身装配完整 TaskService。

    参数:
        checkpointer: 可选的共享检查点，用于验证多轮会话。

    返回值:
        包含真实路由与编排图、但不访问网络的任务服务。
    """
    summary_model = _FakeSummaryModel()
    router_model = _FakeRouterModel()
    checkpointer = checkpointer or create_checkpointer()

    deep_agent = DeepAgentAdapter(runtime=_FakeDeepAgentRuntime())
    with patch("agent_app.workflows.create_chat_model", return_value=summary_model):
        workflows = create_workflows(
            settings=SimpleNamespace(openai=SimpleNamespace(summary_model="summary-test-model")),
            storage=FakeObjectStorage(),
        )
    registry = ExecutorRegistry(
        workflows,
        {
            "solution_planning": ExecutorDefinition(
                mode=SelectedMode.DEEP_AGENT,
                description="Create implementation plans",
                executor=deep_agent,
                is_default=True,
            )
        },
    )
    router = TaskRouter(registry=registry, model=router_model)

    graph = build_orchestration_graph(
        router=router,
        registry=registry,
        checkpointer=checkpointer,
    )

    return TaskService(
        graph=graph,
        registry=registry,
        task_timeout_seconds=10,
    )


@pytest.fixture
def acceptance_service() -> TaskService:
    """创建每个验收用例独享的完整任务服务。

    返回值:
        使用全新内存检查点的任务服务。
    """
    return _build_service()


@pytest.mark.asyncio
async def test_auto_summary_routes_to_workflow(acceptance_service) -> None:
    response = await acceptance_service.invoke(TaskRequest(message="请总结：Alpha Beta"))
    assert response.execution.selected_mode == SelectedMode.WORKFLOW
    assert set(response.result) == {"summary", "key_points"}


@pytest.mark.asyncio
async def test_auto_open_ended_routes_to_deep_agent(acceptance_service) -> None:
    response = await acceptance_service.invoke(
        TaskRequest(message="为一个新产品制定分阶段发布方案")
    )
    assert response.execution.selected_mode == SelectedMode.DEEP_AGENT
    assert response.execution.agent_type == "solution_planning"
    assert set(response.result) == {"answer"}


@pytest.mark.asyncio
async def test_explicit_workflow_override(acceptance_service) -> None:
    response = await acceptance_service.invoke(
        TaskRequest(
            message="do something",
            execution_mode=ExecutionMode.WORKFLOW,
            task_type="summary",
        )
    )
    assert response.execution.selected_mode == SelectedMode.WORKFLOW
    assert response.execution.task_type == "summary"


@pytest.mark.asyncio
async def test_same_thread_follow_up_sees_accumulated_messages(acceptance_service) -> None:
    await acceptance_service.invoke(TaskRequest(message="总结第一段", thread_id="conv-1"))
    state_after = await acceptance_service._graph.aget_state(
        {"configurable": {"thread_id": "conv-1"}}
    )
    assert len(state_after.values.get("messages", [])) == 1

    await acceptance_service.invoke(TaskRequest(message="总结第二段", thread_id="conv-1"))
    state_after_2 = await acceptance_service._graph.aget_state(
        {"configurable": {"thread_id": "conv-1"}}
    )
    assert len(state_after_2.values.get("messages", [])) == 2


@pytest.mark.asyncio
async def test_different_threads_are_isolated(acceptance_service) -> None:
    await acceptance_service.invoke(TaskRequest(message="总结", thread_id="thread-x"))
    await acceptance_service.invoke(TaskRequest(message="总结", thread_id="thread-y"))
    state_x = await acceptance_service._graph.aget_state(
        {"configurable": {"thread_id": "thread-x"}}
    )
    state_y = await acceptance_service._graph.aget_state(
        {"configurable": {"thread_id": "thread-y"}}
    )
    assert len(state_x.values.get("messages", [])) == 1
    assert len(state_y.values.get("messages", [])) == 1


@pytest.mark.asyncio
async def test_stream_success_ends_with_task_completed(acceptance_service) -> None:
    events = [e async for e in acceptance_service.stream(TaskRequest(message="总结"))]
    assert events[0].type is EventType.TASK_STARTED
    assert events[-1].type is EventType.TASK_COMPLETED


@pytest.mark.asyncio
async def test_deep_agent_stream_exposes_agent_type_and_executor_node(
    acceptance_service,
) -> None:
    events = [
        event
        async for event in acceptance_service.stream(
            TaskRequest(
                message="制定发布计划",
                execution_mode=ExecutionMode.DEEP_AGENT,
            )
        )
    ]

    route = next(event for event in events if event.type is EventType.ROUTE_SELECTED)
    node = next(event for event in events if event.type is EventType.NODE_STARTED)
    completed = events[-1]
    assert route.data["task_type"] is None
    assert route.data["agent_type"] == "solution_planning"
    assert node.data["node"] == "deep_agent.solution_planning"
    assert completed.data["agent_type"] == "solution_planning"


@pytest.mark.asyncio
async def test_explicit_unregistered_workflow_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    service = _build_service()
    from agent_app.errors import AppError, ErrorCode

    with pytest.raises(AppError) as error:
        await service.invoke(
            TaskRequest(
                message="x",
                execution_mode=ExecutionMode.WORKFLOW,
                task_type="translate",
            )
        )
    assert error.value.code is ErrorCode.INVALID_TASK_TYPE


@pytest.mark.asyncio
async def test_explicit_pdf_workflow_renders_default_page(monkeypatch) -> None:
    from agent_app.tools.pdf import io as pdf_io

    service = _build_service()
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Page 1")
    document.new_page().insert_text((72, 72), "Page 2")
    content = document.tobytes()
    document.close()

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.content = body

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(pdf_io.httpx, "get", lambda url, timeout=None: _FakeResponse(content))

    response = await service.invoke(
        TaskRequest(
            message="https://example.com/sample.pdf",
            execution_mode=ExecutionMode.WORKFLOW,
            task_type="pdf_to_image",
        )
    )

    assert response.execution.selected_mode is SelectedMode.WORKFLOW
    assert response.execution.task_type == "pdf_to_image"
    assert response.result["page_count"] == 2
    assert response.result["images"]
    assert response.result["images"][0]["url"].startswith("https://fake-s3.test/")
