"""Deep Agent 包级目录测试。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import SecretStr

from agent_app.deep_agents import catalog as catalog_mod
from agent_app.deep_agents import create_agents
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.tasks import SelectedMode


class _FakeRuntime:
    """记录输入并产生固定回答的 Deep Agent runtime。"""

    def __init__(self) -> None:
        self.input_received = None

    async def astream(self, input, config, *, stream_mode, **kwargs):
        self.input_received = input
        yield ("messages", (AIMessageChunk(content="执行方案"), {}))


def _patch_builders(monkeypatch, planning_runtime=None, info_price_runtime=None):
    """替换两个 agent 构建函数并记录调用参数。"""
    planning_calls: list[dict] = []
    info_price_calls: list[dict] = []

    def fake_planning(**kwargs):
        planning_calls.append(kwargs)
        return planning_runtime or _FakeRuntime()

    def fake_info_price(**kwargs):
        info_price_calls.append(kwargs)
        return info_price_runtime or _FakeRuntime()

    monkeypatch.setattr(catalog_mod, "create_solution_planning_agent", fake_planning)
    monkeypatch.setattr(catalog_mod, "create_info_price_agent", fake_info_price)
    return planning_calls, info_price_calls


def _settings(**overrides) -> SimpleNamespace:
    from agent_app.config import S3Config, ZhipuWebConfig

    openai_fields = {
        "solution_planning_model": None,
        "info_price_model": None,
        **overrides,
    }
    return SimpleNamespace(
        openai=SimpleNamespace(**openai_fields),
        zhipu_web=ZhipuWebConfig(),
        s3=S3Config(),
    )


@pytest.mark.asyncio
async def test_create_agents_registers_both_with_single_default(
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
    planning_calls, info_price_calls = _patch_builders(monkeypatch, runtime, runtime)

    agents = create_agents(
        settings=_settings(
            solution_planning_model="planning-specialized-model",
            info_price_model="info-price-specialized-model",
        ),
        checkpointer=object(),
        skill_root=tmp_path,
    )

    # 两个 agent 均注册；恰一个默认（solution_planning）。
    assert set(agents) == {"solution_planning", "info_price"}
    assert agents["solution_planning"].is_default is True
    assert agents["info_price"].is_default is False
    for definition in agents.values():
        assert definition.mode is SelectedMode.DEEP_AGENT
        assert definition.description

    # 各自选择专用模型。
    assert selected_models == [
        "planning-specialized-model",
        "info-price-specialized-model",
    ]
    assert planning_calls[0]["model"] is agent_model
    assert info_price_calls[0]["model"] is agent_model

    # 执行器仍走 DeepAgentAdapter 契约。
    current_message = HumanMessage(content="制定发布计划")
    result = await agents["solution_planning"].executor.run(
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


def test_create_agents_passes_storage_and_web_tools(monkeypatch, tmp_path) -> None:
    """对象存储与搜索/阅读工具透传给对应 agent。"""
    monkeypatch.setattr(
        catalog_mod, "create_chat_model", lambda settings, *, model_name=None: object()
    )
    planning_calls, info_price_calls = _patch_builders(monkeypatch)

    storage = object()
    monkeypatch.setattr(catalog_mod, "create_object_storage", lambda config: storage)
    web_tools = [object(), object()]
    monkeypatch.setattr(
        catalog_mod, "create_web_agent_tools", lambda config: web_tools
    )

    settings = _settings()
    settings.zhipu_web.api_key = SecretStr("sk-zhipu")  # type: ignore[attr-defined]
    create_agents(settings=settings, checkpointer=object(), skill_root=tmp_path)

    assert planning_calls[0]["tools"] is web_tools
    assert info_price_calls[0]["storage"] is storage
    assert info_price_calls[0]["web_tools"] is web_tools


def test_create_agents_without_zhipu_key_omits_web_tools(monkeypatch, tmp_path) -> None:
    """zhipu_web.api_key 未配置（空占位）时不注入搜索/阅读工具。"""
    monkeypatch.setattr(
        catalog_mod, "create_chat_model", lambda settings, *, model_name=None: object()
    )
    planning_calls, info_price_calls = _patch_builders(monkeypatch)

    create_agents(settings=_settings(), checkpointer=object(), skill_root=tmp_path)

    assert planning_calls[0]["tools"] is None
    assert info_price_calls[0]["web_tools"] is None
