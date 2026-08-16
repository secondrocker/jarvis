"""信息价 agent 构造的单元测试（monkeypatch harness）。"""

from agent_app.deep_agents import info_price as info_price_mod
from agent_app.deep_agents.info_price.subagents import build_info_price_subagents


class _WebTool:
    """带名字的最小工具替身。"""

    def __init__(self, name: str) -> None:
        self.name = name


def test_subagents_split_by_function_with_minimal_tools() -> None:
    """子代理按职能划分：researcher 拿 web 工具，analyst 拿图表工具。"""
    subagents = build_info_price_subagents(
        web_tools=[_WebTool("web_search"), _WebTool("web_fetch")],
        chart_tools=[_WebTool("render_chart")],
    )
    assert [item["name"] for item in subagents] == ["researcher", "analyst"]

    researcher, analyst = subagents
    assert [tool.name for tool in researcher["tools"]] == ["web_search", "web_fetch"]
    # researcher 复用主 agent 的信息价技能（网站清单在 references 下）。
    assert researcher["skills"] == ["/skills/info-price/"]
    assert researcher["description"]
    assert researcher["system_prompt"]

    assert [tool.name for tool in analyst["tools"]] == ["render_chart"]
    assert "skills" not in analyst  # 无需站点知识
    assert analyst["description"]
    assert analyst["system_prompt"]

    # 两个子代理均不指定 model（继承主代理）。
    assert "model" not in researcher
    assert "model" not in analyst


def test_subagents_degrade_without_tools() -> None:
    """web/chart 工具缺失时子代理仍构造（降级为无工具）。"""
    subagents = build_info_price_subagents()
    assert [item["name"] for item in subagents] == ["researcher", "analyst"]
    assert subagents[0]["tools"] == []
    assert subagents[1]["tools"] == []


def test_create_info_price_agent_forwards_configuration(monkeypatch, tmp_path) -> None:
    """agent 个性配置（prompt/技能源/子代理）完整透传给公共装配。"""
    build_calls = []

    def fake_build_deep_agent(**kwargs):
        build_calls.append(kwargs)
        return object()

    monkeypatch.setattr(info_price_mod, "build_deep_agent", fake_build_deep_agent)

    web_client = object()
    storage = object()
    web_tools = [_WebTool("web_search")]
    chart_tools = [_WebTool("render_chart")]
    monkeypatch.setattr(info_price_mod, "create_web_agent_tools", lambda client: web_tools)
    monkeypatch.setattr(info_price_mod, "create_chart_agent_tools", lambda client: chart_tools)

    model = object()
    checkpointer = object()
    runtime = info_price_mod.create_info_price_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=tmp_path,
        web_client=web_client,
        storage=storage,
    )
    assert runtime is not None

    call = build_calls[0]
    assert call["model"] is model
    assert call["checkpointer"] is checkpointer
    assert call["skill_root"] == tmp_path
    assert call["skill_sources"] == [("/skills/info-price/", "Info Price")]
    assert "信息价" in call["system_prompt"]
    assert [item["name"] for item in call["subagents"]] == ["researcher", "analyst"]
