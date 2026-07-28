"""TaskRouter 优先级与 LLM 辅助路由的单元测试。"""

import httpx
import pytest
from openai import APITimeoutError

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.router import TaskRouter
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest


class _FakeRoutingRunnable:
    """返回受控 LLMRouteDecision 的异步可运行对象。"""

    def __init__(self, decision, error=None):
        self.decision = decision
        self.error = error

    async def ainvoke(self, input_value):
        if self.error is not None:
            raise self.error
        return self.decision


class _FakeRouterModel:
    """记录调用并返回已绑定可运行对象的 LangChain 兼容替身。"""

    def __init__(self, decision, error=None):
        self.calls = []
        self._decision = decision
        self._error = error
        self._schema = None
        self.runnable = _FakeRoutingRunnable(decision, error)

    def with_structured_output(self, schema):
        self.calls.append(schema)
        self._schema = schema
        return self.runnable


def _registry():
    """构建只包含摘要工作流的注册表。"""
    from agent_app.orchestration.registry import WorkflowRegistry

    class _FakeWorkflow:
        async def ainvoke(self, input, config=None):
            return {"summary": "ok"}

    return WorkflowRegistry({"summary": _FakeWorkflow()})


# ---- 显式执行模式测试 ----


@pytest.mark.asyncio
async def test_explicit_workflow_requires_registered_task_type() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.WORKFLOW,
            task_type="summary",
            is_ambiguous=False,
            reason="test",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)

    with pytest.raises(AppError) as caught:
        await router.route(
            TaskRequest(
                message="do it",
                execution_mode=ExecutionMode.WORKFLOW,
                task_type="translate",
            )
        )
    assert caught.value.code is ErrorCode.INVALID_TASK_TYPE


@pytest.mark.asyncio
async def test_explicit_workflow_with_registered_task_type_no_llm() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.WORKFLOW,
            task_type="summary",
            is_ambiguous=False,
            reason="test",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(
        TaskRequest(message="do it", execution_mode=ExecutionMode.WORKFLOW, task_type="summary")
    )
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert model.calls == []


@pytest.mark.asyncio
async def test_explicit_deep_agent_ignores_task_type_no_llm() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=False,
            reason="test",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(
        TaskRequest(message="plan", execution_mode=ExecutionMode.DEEP_AGENT, task_type="summary")
    )
    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.task_type is None
    assert model.calls == []


# ---- 注册表与确定性规则测试 ----


@pytest.mark.asyncio
async def test_registered_task_type_wins_without_llm() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=True,
            reason="should not be called",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="text", task_type="summary"))
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert model.calls == []


@pytest.mark.parametrize("phrase", ["总结", "摘要", "概括", "summarize", "summary", "SUMMARY"])
@pytest.mark.asyncio
async def test_deterministic_summary_phrases_route_without_llm(phrase) -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=True,
            reason="should not be called",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message=f"请帮我{phrase}这段文字"))
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert model.calls == []


@pytest.mark.asyncio
async def test_broad_word_does_not_trigger_deterministic_rule() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=False,
            reason="open-ended analysis request",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="请分析这段数据"))
    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert len(model.calls) == 1


# ---- LLM 回退测试 ----


@pytest.mark.asyncio
async def test_llm_selects_workflow_with_registered_type() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.WORKFLOW,
            task_type="summary",
            is_ambiguous=False,
            reason="clear summary intent",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="condense the passage"))
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert result.reason == "clear summary intent"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_llm_selects_workflow_with_unregistered_type_falls_back() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.WORKFLOW,
            task_type="translate",
            is_ambiguous=False,
            reason="translation request",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="translate this"))
    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.task_type is None
    assert "translate" not in result.reason


@pytest.mark.asyncio
async def test_ambiguous_llm_decision_falls_back_to_deep_agent() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=True,
            reason="unclear intent",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="帮我想想下一步"))
    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.task_type is None


@pytest.mark.asyncio
async def test_llm_timeout_becomes_upstream_unavailable_not_ambiguous() -> None:
    model = _FakeRouterModel(
        decision=None,
        error=APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
    )
    router = TaskRouter(registry=_registry(), model=model)
    with pytest.raises(AppError) as error:
        await router.route(TaskRequest(message="do something"))
    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE


@pytest.mark.asyncio
async def test_task_type_is_normalized_before_registry_lookup() -> None:
    from agent_app.orchestration.schemas import LLMRouteDecision

    model = _FakeRouterModel(
        LLMRouteDecision(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=None,
            is_ambiguous=True,
            reason="test",
        )
    )
    router = TaskRouter(registry=_registry(), model=model)
    result = await router.route(TaskRequest(message="text", task_type="  Summary  "))
    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.task_type == "summary"
    assert model.calls == []
