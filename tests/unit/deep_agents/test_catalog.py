"""Deep Agent 包级目录测试。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent_app.deep_agents import catalog as catalog_mod
from agent_app.deep_agents import create_agents
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.tasks import SelectedMode


class _FakeRuntime:
    """记录输入并产生固定回答的 Deep Agent runtime。"""

    def __init__(self) -> None:
        self.input_received = None

    async def astream(self, input, config, *, stream_mode):
        self.input_received = input
        yield ("messages", (AIMessageChunk(content="执行方案"), {}))


@pytest.mark.asyncio
async def test_create_agents_returns_default_solution_planning_executor(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _FakeRuntime()
    agent_model = object()
    selected_models = []

    def fake_create_chat_model(settings, *, model_name=None):
        selected_models.append(model_name)
        return agent_model

    monkeypatch.setattr(catalog_mod, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(
        catalog_mod,
        "create_restricted_deep_agent",
        lambda **kwargs: runtime if kwargs["model"] is agent_model else None,
    )

    agents = create_agents(
        settings=SimpleNamespace(
            openai=SimpleNamespace(solution_planning_model="planning-specialized-model")
        ),
        checkpointer=object(),
        skill_root=tmp_path,
    )

    assert set(agents) == {"solution_planning"}
    definition = agents["solution_planning"]
    assert definition.mode is SelectedMode.DEEP_AGENT
    assert definition.description
    assert definition.is_default is True
    assert selected_models == ["planning-specialized-model"]

    current_message = HumanMessage(content="制定发布计划")
    result = await definition.executor.run(
        ExecutionContext(
            message="制定发布计划",
            messages=[current_message],
            parameters={},
            config={"configurable": {"thread_id": "thread-1"}},
            emit=lambda _: None,
        )
    )

    assert result == {"answer": "执行方案"}
    assert runtime.input_received == {"messages": [current_message]}
