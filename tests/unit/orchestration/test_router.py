"""TaskRouter 的统一 workflow/agent 路由测试。"""

from typing import Any

import httpx
import pytest
from openai import APITimeoutError

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext, ExecutorDefinition
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.orchestration.schemas import LLMRouteDecision
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest


class _FakeExecutor:
    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        return {"message": context.message}


class _FakeRoutingRunnable:
    def __init__(self, decision, error=None):
        self.decision = decision
        self.error = error
        self.inputs = []

    async def ainvoke(self, input_value):
        self.inputs.append(input_value)
        if self.error is not None:
            raise self.error
        return self.decision


class _FakeRouterModel:
    def __init__(self, decision, error=None):
        self.calls = []
        self.runnable = _FakeRoutingRunnable(decision, error)

    def with_structured_output(self, schema):
        self.calls.append(schema)
        return self.runnable


def _definition(
    mode: SelectedMode,
    description: str,
    *,
    is_default: bool = False,
) -> ExecutorDefinition:
    return ExecutorDefinition(
        mode=mode,
        description=description,
        executor=_FakeExecutor(),
        is_default=is_default,
    )


def _registry() -> ExecutorRegistry:
    return ExecutorRegistry(
        {"summary": _definition(SelectedMode.WORKFLOW, "Structured text summary")},
        {
            "solution_planning": _definition(
                SelectedMode.DEEP_AGENT,
                "Implementation and delivery planning",
                is_default=True,
            ),
            "research": _definition(
                SelectedMode.DEEP_AGENT,
                "Evidence-based research",
            ),
        },
    )


def _llm_decision(
    mode: SelectedMode,
    executor_type: str | None,
    *,
    ambiguous: bool = False,
) -> LLMRouteDecision:
    return LLMRouteDecision(
        selected_mode=mode,
        executor_type=executor_type,
        is_ambiguous=ambiguous,
        reason="model decision",
    )


@pytest.mark.asyncio
async def test_explicit_workflow_routes_without_llm() -> None:
    model = _FakeRouterModel(None)
    result = await TaskRouter(registry=_registry(), model=model).route(
        TaskRequest(
            message="do it",
            execution_mode=ExecutionMode.WORKFLOW,
            task_type=" Summary ",
        )
    )

    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.executor_type == "summary"
    assert model.calls == []


@pytest.mark.asyncio
async def test_explicit_deep_agent_uses_named_agent_without_llm() -> None:
    model = _FakeRouterModel(None)
    result = await TaskRouter(registry=_registry(), model=model).route(
        TaskRequest(
            message="research it",
            execution_mode=ExecutionMode.DEEP_AGENT,
            agent_type=" Research ",
        )
    )

    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.executor_type == "research"
    assert model.calls == []


@pytest.mark.asyncio
async def test_explicit_deep_agent_without_name_uses_default() -> None:
    result = await TaskRouter(registry=_registry(), model=_FakeRouterModel(None)).route(
        TaskRequest(message="plan", execution_mode=ExecutionMode.DEEP_AGENT)
    )

    assert result.selected_mode is SelectedMode.DEEP_AGENT
    assert result.executor_type == "solution_planning"


@pytest.mark.asyncio
async def test_auto_named_targets_are_validated_and_route_without_llm() -> None:
    model = _FakeRouterModel(None)
    router = TaskRouter(registry=_registry(), model=model)

    workflow = await router.route(TaskRequest(message="x", task_type="summary"))
    agent = await router.route(TaskRequest(message="x", agent_type="research"))

    assert workflow.executor_type == "summary"
    assert agent.executor_type == "research"
    assert model.calls == []


@pytest.mark.parametrize(
    ("task_request", "code"),
    [
        (TaskRequest(message="x", task_type="missing"), ErrorCode.INVALID_TASK_TYPE),
        (TaskRequest(message="x", agent_type="missing"), ErrorCode.INVALID_AGENT_TYPE),
    ],
)
@pytest.mark.asyncio
async def test_auto_unknown_named_target_is_rejected(task_request, code) -> None:
    with pytest.raises(AppError) as error:
        await TaskRouter(registry=_registry(), model=_FakeRouterModel(None)).route(task_request)

    assert error.value.code is code


@pytest.mark.asyncio
async def test_summary_phrase_routes_without_llm() -> None:
    model = _FakeRouterModel(None)
    result = await TaskRouter(registry=_registry(), model=model).route(
        TaskRequest(message="请总结这段文字")
    )

    assert result.selected_mode is SelectedMode.WORKFLOW
    assert result.executor_type == "summary"
    assert model.calls == []


@pytest.mark.parametrize(
    ("decision", "expected_mode", "expected_type"),
    [
        (
            _llm_decision(SelectedMode.WORKFLOW, "summary"),
            SelectedMode.WORKFLOW,
            "summary",
        ),
        (
            _llm_decision(SelectedMode.DEEP_AGENT, "research"),
            SelectedMode.DEEP_AGENT,
            "research",
        ),
        (
            _llm_decision(SelectedMode.WORKFLOW, "missing"),
            SelectedMode.DEEP_AGENT,
            "solution_planning",
        ),
        (
            _llm_decision(SelectedMode.DEEP_AGENT, None, ambiguous=True),
            SelectedMode.DEEP_AGENT,
            "solution_planning",
        ),
    ],
)
@pytest.mark.asyncio
async def test_llm_selection_and_safe_fallback(decision, expected_mode, expected_type) -> None:
    model = _FakeRouterModel(decision)
    result = await TaskRouter(registry=_registry(), model=model).route(
        TaskRequest(message="analyze this request")
    )

    assert result.selected_mode is expected_mode
    assert result.executor_type == expected_type


@pytest.mark.asyncio
async def test_llm_prompt_lists_registered_capability_descriptions() -> None:
    model = _FakeRouterModel(_llm_decision(SelectedMode.DEEP_AGENT, "research"))
    await TaskRouter(registry=_registry(), model=model).route(TaskRequest(message="investigate"))

    prompt_text = "\n".join(
        str(message.content) for message in model.runnable.inputs[0].to_messages()
    )
    assert "summary [workflow]: Structured text summary" in prompt_text
    assert "research [deep_agent]: Evidence-based research" in prompt_text
    assert "solution_planning [deep_agent]: Implementation and delivery planning" in prompt_text


@pytest.mark.asyncio
async def test_llm_timeout_becomes_upstream_unavailable() -> None:
    model = _FakeRouterModel(
        decision=None,
        error=APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
    )

    with pytest.raises(AppError) as error:
        await TaskRouter(registry=_registry(), model=model).route(
            TaskRequest(message="do something")
        )

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
