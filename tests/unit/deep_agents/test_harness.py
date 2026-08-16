"""harness.build_deep_agent 的单元测试（monkeypatch create_deep_agent）。"""

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_app.deep_agents import harness
from agent_app.deep_agents import harness as harness_module


class _CreateCallRecorder:
    """记录 create_deep_agent 调用参数并返回占位运行时。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return object()


def _model() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="ok")])


def test_registers_restricted_profile_once(monkeypatch) -> None:
    recorder = _CreateCallRecorder()
    monkeypatch.setattr(harness_module, "create_deep_agent", recorder)
    register_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        harness_module,
        "register_harness_profile",
        lambda key, profile: register_calls.append((key, profile)),
    )
    monkeypatch.setattr(harness_module, "_PROFILE_REGISTERED", False)

    for _ in range(2):
        harness.build_deep_agent(
            model=_model(),
            checkpointer=MemorySaver(),
            skill_root=Path("/tmp/skills"),
            skill_sources=[("/skills/x/", "X")],
        )

    # 两次构建只注册一次受限 profile，且内容排除 execute、禁用通用子代理。
    assert len(register_calls) == 1
    key, profile = register_calls[0]
    assert key == "openai"
    assert "execute" in profile.excluded_tools
    assert profile.general_purpose_subagent.enabled is False


def test_forwards_agent_specific_configuration(monkeypatch) -> None:
    recorder = _CreateCallRecorder()
    monkeypatch.setattr(harness_module, "create_deep_agent", recorder)
    monkeypatch.setattr(harness_module, "_PROFILE_REGISTERED", True)

    subagent = {
        "name": "researcher",
        "description": "查询数据",
        "system_prompt": "你是数据查询专员。",
    }
    harness.build_deep_agent(
        model=_model(),
        checkpointer=MemorySaver(),
        skill_root=Path("/tmp/skills"),
        skill_sources=[("/skills/info-price/", "Info Price")],
        tools=[],
        system_prompt="你是信息价分析助手。",
        subagents=[subagent],
    )

    call = recorder.calls[0]
    assert call["system_prompt"] == "你是信息价分析助手。"
    assert call["subagents"] == [subagent]
    assert call["skills"] == [("/skills/info-price/", "Info Price")]
    # 空工具列表归一为 None（交由 SDK 决定默认工具集）。
    assert call["tools"] is None
    # /skills/ 前缀路由到只读磁盘，且 permissions 拒绝对技能目录写入。
    backend = call["backend"]
    assert type(backend).__name__ == "CompositeBackend"
    permission = call["permissions"][0]
    assert permission.operations == ["write"]
    assert permission.paths == ["/skills/**"]
    assert permission.mode == "deny"
    assert call["checkpointer"] is not None


def test_empty_tools_normalized_to_none(monkeypatch) -> None:
    recorder = _CreateCallRecorder()
    monkeypatch.setattr(harness_module, "create_deep_agent", recorder)
    monkeypatch.setattr(harness_module, "_PROFILE_REGISTERED", True)

    harness.build_deep_agent(
        model=_model(),
        checkpointer=MemorySaver(),
        skill_root=Path("/tmp/skills"),
        skill_sources=[("/skills/x/", "X")],
        tools=[],
    )
    assert recorder.calls[0]["tools"] is None
