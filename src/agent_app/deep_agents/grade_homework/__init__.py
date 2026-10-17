"""批改作业 Deep Agent：把学生作业照片批改成结构化预批结果。

执行链：``GradeHomeworkExecutor`` 下载并注入图片 → 主代理（视觉模型）
识别分题判分、用确定性工具复算 → ``error_analyst`` / ``feedback_writer``
子代理做错因与评语 → 服务端校验契约并裁定人工复核标记。
"""

from pathlib import Path

from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.deep_agents.grade_homework.executor import GradeHomeworkExecutor
from agent_app.deep_agents.grade_homework.prompts import GRADE_HOMEWORK_SYSTEM_PROMPT
from agent_app.deep_agents.grade_homework.schemas import (
    GradeInput,
    GradeResult,
    QuestionItem,
)
from agent_app.deep_agents.grade_homework.skill_sources import SKILL_SOURCES
from agent_app.deep_agents.grade_homework.subagents import build_grade_homework_subagents
from agent_app.deep_agents.grade_homework.tools import create_grade_agent_tools
from agent_app.deep_agents.harness import build_deep_agent
from agent_app.deep_agents.protocols import DeepAgentRuntime

__all__ = [
    "GradeHomeworkExecutor",
    "GradeInput",
    "GradeResult",
    "QuestionItem",
    "build_grade_homework_subagents",
    "create_grade_agent_tools",
    "create_grade_homework_agent",
]

# 主图对 task 工具的调用上限：一次完整批改最多用 error_analyst 与
# feedback_writer 各一次，留一次余量，防止主代理反复派生子代理空转。
_TASK_RUN_LIMIT = 3


def create_grade_homework_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> DeepAgentRuntime:
    """创建批改作业 agent 运行时。

    参数:
        model: 支持 vision 输入的批改模型（识别图片并判分）。
        checkpointer: 保存多轮会话状态的检查点存储。
        skill_root: 项目内 Deep Agent 技能根目录。

    返回值:
        绑定批改技能、确定性工具与两个职能子代理的受限 Deep Agent 运行时。
    """
    return build_deep_agent(
        model=model,
        checkpointer=checkpointer,
        skill_root=skill_root,
        skill_sources=SKILL_SOURCES,
        tools=create_grade_agent_tools(),
        system_prompt=GRADE_HOMEWORK_SYSTEM_PROMPT,
        subagents=build_grade_homework_subagents(),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="task",
                run_limit=_TASK_RUN_LIMIT,
                exit_behavior="continue",
            ),
        ],
    )
