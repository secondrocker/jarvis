"""方案规划 Deep Agent：把开放式任务拆解为结构化、可验证的执行方案。"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.harness import build_deep_agent
from agent_app.deep_agents.protocols import DeepAgentRuntime

# 本 agent 唯一的技能目录（backend 内虚拟路径 → 磁盘 skill_root 下同名子目录）。
_SKILL_SOURCES: list[tuple[str, str]] = [("/skills/solution-planning/", "Solution Planning")]


def create_solution_planning_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
    tools: list[BaseTool | Callable[..., Any]] | None = None,
) -> DeepAgentRuntime:
    """创建方案规划 agent 运行时。

    参数:
        model: agent 使用的聊天模型。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能根目录。
        tools: 可选的额外工具（如 Web 搜索/抓取）。

    返回值:
        绑定 solution_planning 技能的受限 Deep Agent 运行时。
    """
    return build_deep_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=skill_root,
        skill_sources=_SKILL_SOURCES,
        tools=tools,
    )
