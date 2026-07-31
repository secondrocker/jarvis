"""将固定 LangGraph 工作流适配为统一执行器。"""

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.executors import ExecutionContext

PrepareInput = Callable[[ExecutionContext], dict[str, Any]]


class WorkflowRuntime(Protocol):
    """固定工作流需要实现的最小异步调用契约。"""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """执行工作流并返回完整状态。"""
        ...


class WorkflowExecutor:
    """负责工作流输入转换、执行和公开结果提取。"""

    def __init__(
        self,
        *,
        workflow: WorkflowRuntime,
        prepare_input: PrepareInput,
        invalid_parameters_message: str,
    ) -> None:
        """保存工作流以及该工作流私有的输入适配规则。"""
        self._workflow = workflow
        self._prepare_input = prepare_input
        self._invalid_parameters_message = invalid_parameters_message

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        """校验输入、调用工作流并只返回公开结果。"""
        try:
            workflow_input = self._prepare_input(context)
        except ValidationError as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                self._invalid_parameters_message,
            ) from error

        try:
            result = await self._workflow.ainvoke(workflow_input, context.config)
        except AppError:
            raise
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Workflow execution failed",
            ) from error

        return result.get("result", result) if isinstance(result, dict) else {}
