"""信息价 agent 的职能子代理定义。"""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import SubAgent
from langchain_core.tools import BaseTool

from agent_app.deep_agents.info_price.prompts import ANALYST_DESCRIPTION, ANALYST_SYSTEM_PROMPT


def build_info_price_subagents(
    *,
    chart_tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
) -> list[SubAgent]:
    """构造 analyst 子代理：趋势分析与图表渲染。

    子代理不指定 model（继承主代理），工具最小化：analyst 只拿图表渲染。

    参数:
        chart_tools: render_chart 工具（对象存储未配置时为空）。

    返回值:
        analyst 的 SubAgent 定义列表。
    """
    return [
        {
            "name": "analyst",
            "description": ANALYST_DESCRIPTION,
            "system_prompt": ANALYST_SYSTEM_PROMPT,
            "tools": list(chart_tools) if chart_tools else [],
        },
    ]
