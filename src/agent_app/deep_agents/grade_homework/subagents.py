"""批改作业 agent 的职能子代理定义。

子代理只做纯文本推理（不接触图片）：主代理完成识别与判分后，把逐题
结果作为文本交给它们做错因诊断与评语撰写，避免"再读一遍图"带来的
感知误差与额外视觉调用成本。
"""

from deepagents import SubAgent

from agent_app.deep_agents.grade_homework.prompts import (
    ERROR_ANALYST_DESCRIPTION,
    ERROR_ANALYST_SYSTEM_PROMPT,
    FEEDBACK_WRITER_DESCRIPTION,
    FEEDBACK_WRITER_SYSTEM_PROMPT,
)
from agent_app.deep_agents.grade_homework.schemas import DiagnosisResult, FeedbackReport
from agent_app.deep_agents.grade_homework.skill_sources import SKILL_CONTAINER


def build_grade_homework_subagents() -> list[SubAgent]:
    """构造错因诊断与评语撰写两个子代理。

    两者都绑定同一个技能容器（按描述命中后自行读取对应 SKILL.md），
    不继承主代理的工具（``tools=[]``），并用结构化输出契约把结果回传
    给主代理，便于主代理原样并入最终批改结果。

    返回值:
        error_analyst / feedback_writer 的 SubAgent 定义列表。
    """
    return [
        {
            "name": "error_analyst",
            "description": ERROR_ANALYST_DESCRIPTION,
            "system_prompt": ERROR_ANALYST_SYSTEM_PROMPT,
            "tools": [],
            "skills": [SKILL_CONTAINER],
            "response_format": DiagnosisResult,
        },
        {
            "name": "feedback_writer",
            "description": FEEDBACK_WRITER_DESCRIPTION,
            "system_prompt": FEEDBACK_WRITER_SYSTEM_PROMPT,
            "tools": [],
            "skills": [SKILL_CONTAINER],
            "response_format": FeedbackReport,
        },
    ]
