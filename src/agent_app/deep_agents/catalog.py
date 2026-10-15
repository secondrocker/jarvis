"""创建应用可用的全部 Deep Agent。"""

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.config import Settings
from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.info_price import create_info_price_agent
from agent_app.deep_agents.solution_planning import create_solution_planning_agent
from agent_app.infrastructure.llm import create_chat_model
from agent_app.infrastructure.storage import create_object_storage
from agent_app.orchestration.executors import ExecutorDefinition
from agent_app.schemas.tasks import SelectedMode


def create_agents(
    *,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> dict[str, ExecutorDefinition]:
    """创建全部 Deep Agent 及其路由元数据。"""
    storage = create_object_storage(settings.s3)
    planning = create_solution_planning_agent(
        model=create_chat_model(
            settings,
            model_name=settings.openai.solution_planning_model,
        ),
        checkpointer=checkpointer,
        skill_root=skill_root,
    )
    info_price = create_info_price_agent(
        model=create_chat_model(
            settings,
            model_name=settings.openai.info_price_model,
        ),
        checkpointer=checkpointer,
        skill_root=skill_root,
        storage=storage,
    )
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
    }
