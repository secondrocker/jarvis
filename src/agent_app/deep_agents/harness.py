"""Deep Agent 的最小公共装配：受限 profile、后端路由与图构建。

每个 agent 有独立的 prompt、工具、子代理与技能，个性配置放在各自的
模块里；本模块只沉淀所有 agent 共享的装配约束。
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    SubAgent,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware import FilesystemPermission
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.protocols import DeepAgentRuntime

# 按规格移除 Shell 工具。`task`（子代理调度）工具则通过禁用自动添加的
# 通用子代理单独移除，因为 harness 会拒绝直接排除 `SubAgentMiddleware`，
# 并抛出 `ValueError`。需要子代理的 agent 显式传入 `subagents=`，此时
# `task` 工具仍然装配，只是不再附带通用子代理条目。
_EXCLUDED_TOOLS = frozenset({"execute"})

# OpenAI 聊天模型解析得到的提供方键（见 infrastructure/llm.py）。
# `create_deep_agent` 会按模型提供方查找 harness profile，因此对于预构建的
# ChatOpenAI，需要在提供方级别注册才能正确匹配。
_PROVIDER_KEY = "openai"

# 技能源在 backend 内的挂载前缀：路由到只读的真实磁盘目录。
_SKILLS_ROUTE = "/skills/"

# profile 只需注册一次：注册表是进程级全局 dict 且重复注册会 additive merge。
_PROFILE_REGISTERED = False


def _ensure_restricted_profile() -> None:
    """幂等注册受限 HarnessProfile（排除 Shell、禁用通用子代理）。"""
    global _PROFILE_REGISTERED
    if _PROFILE_REGISTERED:
        return
    register_harness_profile(
        _PROVIDER_KEY,
        HarnessProfile(
            excluded_tools=_EXCLUDED_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _PROFILE_REGISTERED = True


def build_deep_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
    skill_sources: Sequence[tuple[str, str]],
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    subagents: Sequence[SubAgent] | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
) -> DeepAgentRuntime:
    """按公共约束构建受限 Deep Agent。

    默认装配 TodoListMiddleware（task list）与 SummarizationMiddleware
    （上下文自动压缩），传入 subagents 即启用 `task` 子代理调度工具。

    参数:
        model: Deep Agent 使用的聊天模型。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能根目录（真实磁盘路径）。
        skill_sources: 技能源 ``(backend 路径, 显示标签)`` 元组列表，如
            ``("/skills/solution-planning/", "Solution Planning")``；显式
            标签避免默认按叶子目录名渲染出 "Skills Skills"。
        tools: 可选的额外工具（如 Web 搜索/抓取、图表渲染）；harness 的
            受限 profile 按名字精确排除内置工具，不影响自定义工具。
        system_prompt: agent 个性指令，置于 SDK 基础 prompt 之前。
        subagents: 可选的职能子代理定义；传入后主代理获得 `task` 工具。
        middleware: 可选的额外中间件（如 web 工具限流），置于 SDK 默认
            栈之后装配；主图与子代理图步数预算独立，需限流的图各自注入。

    返回值:
        已禁用 Shell 与通用子代理的 Deep Agent 运行时。
    """
    _ensure_restricted_profile()
    # SkillsMiddleware 只通过 backend API 读文件（无磁盘回退），而
    # StateBackend 只能读到 state 中的 files 通道——必须把 /skills/ 前缀
    # 路由到真实磁盘，技能才会在运行时被加载进 system prompt。
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            _SKILLS_ROUTE: FilesystemBackend(
                root_dir=skill_root,
                virtual_mode=True,
            ),
        },
    )
    return create_deep_agent(
        model=model,
        tools=list(tools) if tools else None,
        system_prompt=system_prompt,
        subagents=list(subagents) if subagents else None,
        middleware=list(middleware) if middleware else None,
        skills=[tuple(source) for source in skill_sources],
        permissions=[
            # /skills/ 路由直通真实磁盘源码目录，必须拒绝写入；
            # 子代理默认继承父级 permissions，无需重复声明。
            FilesystemPermission(
                operations=["write"],
                paths=["/skills/**"],
                mode="deny",
            ),
        ],
        backend=backend,
        checkpointer=checkpointer,
    )
