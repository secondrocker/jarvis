"""仅启用获准能力的受限 Deep Agent 工厂。"""

from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.protocols import DeepAgentRuntime

# 按规格移除 Shell 工具。`task`（子代理调度）工具则通过禁用自动添加的
# 通用子代理单独移除，因为 harness 会拒绝直接排除 `SubAgentMiddleware`，
# 并抛出 `ValueError`。
_EXCLUDED_TOOLS = ("execute",)

# OpenAI 聊天模型解析得到的提供方键（见 infrastructure/llm.py）。
# `create_deep_agent` 会按模型提供方查找 harness profile，因此对于预构建的
# ChatOpenAI，需要在提供方级别注册才能正确匹配。
_RESTRICTED_PROVIDER_KEY = "openai"


def create_restricted_deep_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> DeepAgentRuntime:
    """创建仅包含获准技能和内存能力的运行时。

    参数:
        model: Deep Agent 使用的聊天模型。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能目录。

    返回值:
        已禁用 Shell 与子代理调度能力的 Deep Agent 运行时。
    """
    profile = HarnessProfile(
        excluded_tools=frozenset(_EXCLUDED_TOOLS),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    register_harness_profile(_RESTRICTED_PROVIDER_KEY, profile)

    skill_path = str(skill_root / "solution_planning")

    return create_deep_agent(
        model=model,
        tools=None,
        skills=[skill_path],
        backend=StateBackend(),
        checkpointer=checkpointer,
    )
