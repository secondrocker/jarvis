"""信息价综合 Deep Agent：查询、比价与趋势分析。"""

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.harness import build_deep_agent
from agent_app.deep_agents.info_price.prompts import INFO_PRICE_SYSTEM_PROMPT
from agent_app.deep_agents.info_price.subagents import build_info_price_subagents
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.infrastructure.web_gateway import WebGatewayClient
from agent_app.tools.chart_tools import create_chart_agent_tools
from agent_app.tools.web_tools import create_web_agent_tools

# 本 agent 的技能源（backend 内虚拟路径 → 磁盘 skill_root 下同名子目录）。
_SKILL_SOURCES: list[tuple[str, str]] = [("/skills/info-price/", "Info Price")]


def create_info_price_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
    web_client: WebGatewayClient | None,
    storage: ObjectStorage | None,
) -> DeepAgentRuntime:
    """创建信息价综合 agent 运行时。

    主代理负责编排与撰写报告，researcher 子代理取数，analyst 子代理
    分析出图；网关/对象存储未配置时对应子代理降级（无工具仍可工作，
    如实报告渠道不可用）。

    参数:
        model: agent 使用的聊天模型（子代理继承）。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能根目录。
        web_client: Web 网关客户端；未配置时 researcher 无 web 工具。
        storage: 对象存储；未配置时 analyst 无图表工具（降级表格）。

    返回值:
        绑定 info-price 技能与职能子代理的受限 Deep Agent 运行时。
    """
    return build_deep_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=skill_root,
        skill_sources=_SKILL_SOURCES,
        tools=None,
        system_prompt=INFO_PRICE_SYSTEM_PROMPT,
        subagents=build_info_price_subagents(
            web_tools=create_web_agent_tools(web_client) if web_client else None,
            chart_tools=create_chart_agent_tools(storage),
        ),
    )
