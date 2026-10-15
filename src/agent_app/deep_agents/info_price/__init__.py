"""信息价综合 Deep Agent：查询、比价与趋势分析。"""

from pathlib import Path

from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.harness import build_deep_agent
from agent_app.deep_agents.info_price.prompts import INFO_PRICE_SYSTEM_PROMPT
from agent_app.deep_agents.info_price.subagents import build_info_price_subagents
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.tools.chart_tools import create_chart_agent_tools

# 本 agent 的技能源（backend 内虚拟路径 → 磁盘 skill_root 下同名子目录）。
# 注意：source 须为容器目录，技能为其一级子目录（SDK 按此扫描 SKILL.md）。
_SKILL_SOURCES: list[tuple[str, str]] = [("/skills/info-price/", "Info Price")]

# 主图对 task 工具的调用次数上限：防止主代理反复派生子代理空转。
# 限 2 次覆盖正常流程（analyst 1 次）+ 少量余量。
_TASK_RUN_LIMIT = 2


def create_info_price_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
    storage: ObjectStorage | None,
) -> DeepAgentRuntime:
    """创建信息价综合 agent 运行时。

    主代理负责编排、取数与撰写报告，analyst 子代理分析出图；
    对象存储未配置时 analyst 降级（无图表工具仍可工作，表格呈现）。

    参数:
        model: agent 使用的聊天模型（子代理继承）。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能根目录。
        storage: 对象存储；未配置时 analyst 无图表工具（降级表格）。

    返回值:
        绑定 info-price 技能与 analyst 子代理的受限 Deep Agent 运行时。
    """
    return build_deep_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=skill_root,
        skill_sources=_SKILL_SOURCES,
        tools=None,
        system_prompt=INFO_PRICE_SYSTEM_PROMPT,
        subagents=build_info_price_subagents(
            chart_tools=create_chart_agent_tools(storage),
        ),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="task",
                run_limit=_TASK_RUN_LIMIT,
                exit_behavior="continue",
            ),
        ],
    )
