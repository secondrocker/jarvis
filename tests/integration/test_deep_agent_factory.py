"""Integration test for the restricted deep agent factory."""

from pathlib import Path

from agent_app.deep_agents import factory as factory_mod
from agent_app.deep_agents.factory import create_restricted_deep_agent


def test_factory_passes_only_approved_skill_and_no_external_tools(monkeypatch) -> None:
    """Factory must register only solution_planning skill with no extra tools."""
    captured = {}

    def fake_create_deep_agent(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)

    skill_root = Path("/fake/skills")
    create_restricted_deep_agent(
        model=object(),
        checkpointer=object(),
        skill_root=skill_root,
    )

    kwargs = captured["kwargs"]
    assert kwargs["tools"] is None
    assert kwargs["skills"] == [str(skill_root / "solution_planning")]
    assert kwargs["backend"].__class__.__name__ == "StateBackend"
    assert kwargs["checkpointer"] is not None


def test_factory_excludes_shell_and_subagent_tools(monkeypatch) -> None:
    """The harness profile must exclude execute and task tools."""
    captured = {}

    def fake_create_deep_agent(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(factory_mod, "create_deep_agent", fake_create_deep_agent)

    create_restricted_deep_agent(
        model=object(), checkpointer=object(), skill_root=Path("/fake/skills"),
    )

    profile_label = captured["kwargs"].get("harness_profile")
    assert profile_label == "restricted-demo"
