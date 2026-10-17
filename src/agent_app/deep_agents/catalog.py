"""创建应用可用的全部 Deep Agent。"""

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.config import Settings
from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.grade_homework import (
    GradeHomeworkExecutor,
    create_grade_homework_agent,
)
from agent_app.deep_agents.info_price import create_info_price_agent
from agent_app.deep_agents.solution_planning import create_solution_planning_agent
from agent_app.infrastructure.llm import create_chat_model
from agent_app.infrastructure.storage import create_object_storage
from agent_app.orchestration.executors import ExecutorDefinition
from agent_app.schemas.tasks import SelectedMode
from agent_app.tools.web_search_tools import create_web_agent_tools


def _create_grade_model(settings: Settings) -> BaseChatModel:
    """按批改专用配置创建视觉模型；未配置时回退全局默认模型与端点。"""
    api_key = getattr(settings.openai, "grade_homework_api_key", None)
    return create_chat_model(
        settings,
        model_name=getattr(settings.openai, "grade_homework_model", None),
        base_url=getattr(settings.openai, "grade_homework_base_url", None),
        api_key=api_key.get_secret_value() if api_key is not None else None,
    )


def create_agents(
    *,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> dict[str, ExecutorDefinition]:
    """创建全部 Deep Agent 及其路由元数据。"""
    storage = create_object_storage(settings.s3)
    # 搜索/阅读工具不做用量限制（需求明确要求）；未配置 api_key 时不注入。
    web_tools = create_web_agent_tools(settings.zhipu_web) if settings.zhipu_web.enabled else None
    planning = create_solution_planning_agent(
        model=create_chat_model(
            settings,
            model_name=settings.openai.solution_planning_model,
        ),
        checkpointer=checkpointer,
        skill_root=skill_root,
        tools=web_tools,
    )
    info_price = create_info_price_agent(
        model=create_chat_model(
            settings,
            model_name=settings.openai.info_price_model,
        ),
        checkpointer=checkpointer,
        skill_root=skill_root,
        storage=storage,
        web_tools=web_tools,
    )
    grade_runtime = create_grade_homework_agent(
        model=_create_grade_model(settings),
        checkpointer=checkpointer,
        skill_root=skill_root,
    )
    # getattr 容错：单元测试用 SimpleNamespace 替身 settings，未必带新字段。
    grade_config = getattr(settings, "grade_homework", None)
    return {
        "solution_planning": ExecutorDefinition(
            mode=SelectedMode.DEEP_AGENT,
            description="Create structured implementation and delivery plans",
            executor=DeepAgentAdapter(runtime=planning),
            is_default=True,
        ),
        "info_price": ExecutorDefinition(
            mode=SelectedMode.DEEP_AGENT,
            description=(
                "Query and compare regional construction material info-prices "
                "with trend analysis and charts"
            ),
            executor=DeepAgentAdapter(runtime=info_price),
            is_default=False,
        ),
        "grade_homework": ExecutorDefinition(
            mode=SelectedMode.DEEP_AGENT,
            description=(
                "Grade a homework photo: detect the subject (math/chinese/english), judge every "
                "question step by step, verify math with an exact calculator, diagnose mistakes "
                "and write student feedback"
            ),
            executor=GradeHomeworkExecutor(
                runtime=grade_runtime,
                max_image_mb=getattr(grade_config, "max_image_mb", 8),
                max_image_edge=getattr(grade_config, "max_image_edge", 2048),
                max_pages=getattr(grade_config, "max_pages", 4),
            ),
            is_default=False,
        ),
    }
