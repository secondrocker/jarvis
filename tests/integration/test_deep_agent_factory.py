"""受限 Deep Agent 工厂的集成测试。"""

from pathlib import Path

from agent_app.deep_agents import factory as factory_mod
from agent_app.deep_agents.factory import create_restricted_deep_agent


def test_factory_passes_only_approved_skill_and_no_external_tools(monkeypatch) -> None:
    """工厂只能注册 solution_planning 技能，且不能包含额外工具。"""
    create_calls = []

    def fake_create_deep_agent(*args, **kwargs):
        create_calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(factory_mod, "register_harness_profile", lambda *args, **kwargs: None)

    skill_root = Path("/fake/skills")
    create_restricted_deep_agent(
        model=object(),
        checkpointer=object(),
        skill_root=skill_root,
    )

    kwargs = create_calls[0]
    assert kwargs["tools"] is None
    assert kwargs["skills"] == [str(skill_root / "solution_planning")]
    assert kwargs["backend"].__class__.__name__ == "StateBackend"
    assert kwargs["checkpointer"] is not None


def test_factory_registers_profile_excluding_shell_and_subagent_tools(
    monkeypatch,
) -> None:
    """harness profile 必须为模型提供方移除 execute/task 工具。"""
    registrations = []

    def fake_register(key, profile):
        registrations.append((key, profile))

    monkeypatch.setattr(factory_mod, "register_harness_profile", fake_register)
    monkeypatch.setattr(factory_mod, "create_deep_agent", lambda *args, **kwargs: object())

    create_restricted_deep_agent(
        model=object(),
        checkpointer=object(),
        skill_root=Path("/fake/skills"),
    )

    assert len(registrations) == 1
    key, profile = registrations[0]
    assert key == "openai"
    assert "execute" in profile.excluded_tools
    # 通过禁用自动添加的通用子代理来移除 task 工具。
    assert profile.general_purpose_subagent.enabled is False


def test_factory_passes_custom_tools_through_to_deep_agent(monkeypatch) -> None:
    """显式注入的工具必须原样传给 create_deep_agent，且受限 profile 不受影响。"""
    create_calls = []
    registrations = []

    def fake_create_deep_agent(*args, **kwargs):
        create_calls.append(kwargs)
        return object()

    def fake_register(key, profile):
        registrations.append((key, profile))

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(factory_mod, "register_harness_profile", fake_register)

    web_search = object()
    web_fetch = object()
    create_restricted_deep_agent(
        model=object(),
        checkpointer=object(),
        skill_root=Path("/fake/skills"),
        tools=[web_search, web_fetch],
    )

    assert create_calls[0]["tools"] == [web_search, web_fetch]
    # 自定义工具注入与受限 profile（排除 execute、禁用子代理）共存。
    assert "execute" in registrations[0][1].excluded_tools
    assert registrations[0][1].general_purpose_subagent.enabled is False


def test_factory_normalizes_empty_tools_to_none(monkeypatch) -> None:
    """空工具列表归一为 None，与未注入保持一致语义。"""
    create_calls = []

    def fake_create_deep_agent(*args, **kwargs):
        create_calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(factory_mod, "register_harness_profile", lambda *args, **kwargs: None)

    create_restricted_deep_agent(
        model=object(),
        checkpointer=object(),
        skill_root=Path("/fake/skills"),
        tools=[],
    )

    assert create_calls[0]["tools"] is None
