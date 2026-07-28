"""Integration test for the restricted deep agent factory."""

from pathlib import Path

from agent_app.deep_agents import factory as factory_mod
from agent_app.deep_agents.factory import create_restricted_deep_agent


def test_factory_passes_only_approved_skill_and_no_external_tools(monkeypatch) -> None:
    """Factory must register only solution_planning skill with no extra tools."""
    create_calls = []

    def fake_create_deep_agent(*args, **kwargs):
        create_calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        factory_mod, "register_harness_profile", lambda *args, **kwargs: None
    )

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
    """The harness profile must drop execute/task tools for the model's provider."""
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
    # task tool is removed by disabling the auto-added general-purpose subagent.
    assert profile.general_purpose_subagent.enabled is False
