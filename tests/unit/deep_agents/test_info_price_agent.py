"""信息价 agent 构造的单元测试（monkeypatch harness）。"""

from pathlib import Path

import yaml
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

import agent_app
from agent_app.deep_agents import info_price as info_price_mod
from agent_app.deep_agents.info_price.subagents import build_info_price_subagents


class _ChartTool:
    """带名字的最小工具替身。"""

    def __init__(self, name: str) -> None:
        self.name = name


def test_subagents_minimal_tools() -> None:
    """子代理按职能划分：analyst 拿图表工具。"""
    subagents = build_info_price_subagents(
        chart_tools=[_ChartTool("render_chart")],
    )
    assert [item["name"] for item in subagents] == ["analyst"]

    analyst = subagents[0]
    assert [tool.name for tool in analyst["tools"]] == ["render_chart"]
    assert "skills" not in analyst  # 无需站点知识
    assert analyst["description"]
    assert analyst["system_prompt"]

    # 子代理不指定 model（继承主代理）。
    assert "model" not in analyst


def test_create_info_price_agent_forwards_configuration(monkeypatch, tmp_path) -> None:
    """agent 个性配置（prompt/技能源/子代理）完整透传给公共装配。"""
    build_calls = []

    def fake_build_deep_agent(**kwargs):
        build_calls.append(kwargs)
        return object()

    monkeypatch.setattr(info_price_mod, "build_deep_agent", fake_build_deep_agent)

    storage = object()
    chart_tools = [_ChartTool("render_chart")]
    monkeypatch.setattr(info_price_mod, "create_chart_agent_tools", lambda client: chart_tools)

    model = object()
    checkpointer = object()
    runtime = info_price_mod.create_info_price_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=tmp_path,
        storage=storage,
    )
    assert runtime is not None

    call = build_calls[0]
    assert call["model"] is model
    assert call["checkpointer"] is checkpointer
    assert call["skill_root"] == tmp_path
    assert call["skill_sources"] == [("/skills/info-price/", "Info Price")]
    assert "信息价" in call["system_prompt"]
    assert [item["name"] for item in call["subagents"]] == ["analyst"]

    # 主图装配 task 限流中间件（防主代理无限派生子代理空转）。
    assert isinstance(call["middleware"][0], ToolCallLimitMiddleware)
    assert call["middleware"][0].tool_name == "task"
    assert call["middleware"][0].run_limit == 2


def test_skill_layout_loadable() -> None:
    """技能源容器下必须存在含 SKILL.md 的一级子目录，且 name 与目录名一致。

    SDK 的 SkillsMiddleware 只扫描 source 的一级子目录（source 根下的
    SKILL.md 不会被注入），此测试防止容器化布局被破坏后技能静默失效。
    """
    skills_root = Path(agent_app.__file__).resolve().parent / "skills"
    source_paths = [path for path, _ in info_price_mod._SKILL_SOURCES]

    for source in source_paths:
        container = skills_root / source.removeprefix("/skills/").strip("/")
        skill_dirs = [d for d in container.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()]
        assert skill_dirs, f"{source} 下没有含 SKILL.md 的一级子目录，技能注入将为空"
        for skill_dir in skill_dirs:
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
            assert frontmatter["name"] == skill_dir.name
