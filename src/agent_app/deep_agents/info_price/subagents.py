"""信息价 agent 的职能子代理定义。"""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import SubAgent
from langchain_core.tools import BaseTool

from agent_app.deep_agents.info_price.prompts import (
    ANALYST_DESCRIPTION,
    ANALYST_SYSTEM_PROMPT,
    RESEARCHER_DESCRIPTION,
    RESEARCHER_SYSTEM_PROMPT,
)

# researcher 复用主 agent 的信息价技能（含网站清单 references）。
_RESEARCHER_SKILLS = ["/skills/info-price/"]


def build_info_price_subagents(
    *,
    web_tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    chart_tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
) -> list[SubAgent]:
    """构造按职能划分的子代理：researcher 取数、analyst 分析出图。

    两个子代理均不指定 model（继承主代理），工具各自最小化：
    researcher 只拿 web 搜索/抓取，analyst 只拿图表渲染。

    参数:
        web_tools: web_search/web_fetch 工具（网关未配置时为空）。
        chart_tools: render_chart 工具（对象存储未配置时为空）。

    返回值:
        researcher 与 analyst 的 SubAgent 定义列表。
    """
    return [
        {
            "name": "researcher",
            "description": RESEARCHER_DESCRIPTION,
            "system_prompt": RESEARCHER_SYSTEM_PROMPT,
            "tools": list(web_tools) if web_tools else [],
            "skills": _RESEARCHER_SKILLS,
        },
        {
            "name": "analyst",
            "description": ANALYST_DESCRIPTION,
            "system_prompt": ANALYST_SYSTEM_PROMPT,
            "tools": list(chart_tools) if chart_tools else [],
        },
    ]
