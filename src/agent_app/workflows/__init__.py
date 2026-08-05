"""创建应用可用的全部固定 LangGraph 工作流。"""

from agent_app.config import Settings
from agent_app.infrastructure.llm import create_chat_model
from agent_app.infrastructure.storage import ObjectStorage, create_object_storage
from agent_app.orchestration.executors import (
    ExecutionContext,
    ExecutorDefinition,
)
from agent_app.schemas.tasks import SelectedMode
from agent_app.workflows.adapter import WorkflowExecutor
from agent_app.workflows.pdf_to_image import PdfInput, build_pdf_to_image_graph
from agent_app.workflows.summary import SummaryInput, build_summary_graph


def _prepare_summary_input(context: ExecutionContext) -> dict:
    """把统一执行上下文转换为已校验的摘要输入。"""
    payload = {**context.parameters, "text": context.message}
    return SummaryInput.model_validate(payload).model_dump()


def _prepare_pdf_input(context: ExecutionContext) -> dict:
    """把统一执行上下文转换为已校验的 PDF 转图片输入。

    ``message`` 视为可下载的 PDF URL。
    """
    payload = {**context.parameters, "url": context.message}
    return PdfInput.model_validate(payload).model_dump()


def create_workflows(
    *, settings: Settings, storage: ObjectStorage | None = None
) -> dict[str, ExecutorDefinition]:
    """创建全部固定 workflow 及其路由元数据。

    参数:
        settings: 应用配置。
        storage: 可选的对象存储；未提供时按 settings.s3 构造（PDF 工具上传产物用）。
    """
    summary = WorkflowExecutor(
        workflow=build_summary_graph(
            create_chat_model(settings, model_name=settings.openai.summary_model)
        ),
        prepare_input=_prepare_summary_input,
        invalid_parameters_message="Invalid summary parameters",
    )
    pdf_to_image = WorkflowExecutor(
        workflow=build_pdf_to_image_graph(storage=storage or create_object_storage(settings.s3)),
        prepare_input=_prepare_pdf_input,
        invalid_parameters_message="Invalid PDF parameters",
    )
    return {
        "summary": ExecutorDefinition(
            mode=SelectedMode.WORKFLOW,
            description="Create a structured summary with key points",
            executor=summary,
        ),
        "pdf_to_image": ExecutorDefinition(
            mode=SelectedMode.WORKFLOW,
            description="Render PDF pages to images and return S3 download URLs",
            executor=pdf_to_image,
        ),
    }


__all__ = ["create_workflows"]
