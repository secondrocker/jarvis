"""创建应用可用的全部 Deep Agent。"""

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_app.config import Settings
from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.deep_agents.factory import create_restricted_deep_agent
from agent_app.infrastructure.llm import create_chat_model
from agent_app.orchestration.executors import ExecutorDefinition
from agent_app.schemas.tasks import SelectedMode


def create_agents(
    *,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    skill_root: Path,
) -> dict[str, ExecutorDefinition]:
    """创建全部 Deep Agent 及其路由元数据。"""
    runtime = create_restricted_deep_agent(
        model=create_chat_model(
            settings,
            model_name=settings.solution_planning_model,
        ),
        checkpointer=checkpointer,
        skill_root=skill_root,
    )
    return {
        "solution_planning": ExecutorDefinition(
            mode=SelectedMode.DEEP_AGENT,
            description="Create structured implementation and delivery plans",
            executor=DeepAgentAdapter(runtime=runtime),
            is_default=True,
        )
    }
