"""创建应用可用的全部固定 LangGraph 工作流。"""

import re

from agent_app.config import Settings
from agent_app.infrastructure.llm import create_chat_model
from agent_app.infrastructure.storage import ObjectStorage, create_object_storage
from agent_app.orchestration.executors import (
    ExecutionContext,
    ExecutorDefinition,
)
from agent_app.schemas.tasks import SelectedMode
from agent_app.workflows.adapter import WorkflowExecutor
from agent_app.workflows.grade_homework import GradeInput, build_grade_homework_graph
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


def _prepare_grade_input(context: ExecutionContext) -> dict:
    """把统一执行上下文转换为已校验的批改输入。

    ``message`` 为一个或多个图片 URL(空白/逗号分隔,跨页作业多图一次提交)。
    """
    urls = [u for u in re.split(r"[\s,]+", context.message.strip()) if u]
    payload = {**context.parameters, "urls": urls}
    return GradeInput.model_validate(payload).model_dump()


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
    # getattr 容错: 单元测试用 SimpleNamespace 替身 settings,未必带新字段。
    grade_model_name = getattr(settings.openai, "grade_homework_model", None)
    grade_config = getattr(settings, "grade_homework", None)
    grade_homework = WorkflowExecutor(
        workflow=build_grade_homework_graph(
            create_chat_model(
                settings,
                model_name=grade_model_name,
                base_url=getattr(settings.openai, "grade_homework_base_url", None),
                api_key=(
                    getattr(settings.openai, "grade_homework_api_key", None)
                    and settings.openai.grade_homework_api_key.get_secret_value()
                ),
            ),
            max_image_mb=getattr(grade_config, "max_image_mb", 8),
            max_image_edge=getattr(grade_config, "max_image_edge", 2048),
        ),
        prepare_input=_prepare_grade_input,
        invalid_parameters_message="Invalid grade homework parameters",
    )
    return {
        "grade_homework": ExecutorDefinition(
            mode=SelectedMode.WORKFLOW,
            description="Grade a homework photo: detect subject (math/chinese/english), "
            "check handwritten answers step by step, verify math with an exact calculator",
            executor=grade_homework,
        ),
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
