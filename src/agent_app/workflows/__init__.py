"""创建应用可用的全部固定 LangGraph 工作流。"""

from agent_app.config import Settings
from agent_app.infrastructure.llm import create_chat_model
from agent_app.orchestration.executors import (
    ExecutionContext,
    ExecutorDefinition,
)
from agent_app.schemas.tasks import SelectedMode
from agent_app.workflows.adapter import WorkflowExecutor
from agent_app.workflows.summary import SummaryInput, build_summary_graph


def _prepare_summary_input(context: ExecutionContext) -> dict:
    """把统一执行上下文转换为已校验的摘要输入。"""
    payload = {**context.parameters, "text": context.message}
    return SummaryInput.model_validate(payload).model_dump()


def create_workflows(*, settings: Settings) -> dict[str, ExecutorDefinition]:
    """创建全部固定 workflow 及其路由元数据。"""
    summary = WorkflowExecutor(
        workflow=build_summary_graph(
            create_chat_model(settings, model_name=settings.summary_model)
        ),
        prepare_input=_prepare_summary_input,
        invalid_parameters_message="Invalid summary parameters",
    )
    return {
        "summary": ExecutorDefinition(
            mode=SelectedMode.WORKFLOW,
            description="Create a structured summary with key points",
            executor=summary,
        )
    }


__all__ = ["create_workflows"]
