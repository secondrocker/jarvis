"""用于显式注册任务类型的工作流注册表。"""

from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

from agent_app.errors import AppError, ErrorCode


@runtime_checkable
class Workflow(Protocol):
    """已注册固定工作流的执行契约。"""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """执行工作流并返回结果字典。

        参数:
            input: 工作流初始状态。
            config: 可选的 LangGraph 运行配置。

        返回值:
            工作流执行后的完整状态字典。
        """
        ...


def _normalize(task_type: str) -> str:
    """规范化任务类型，以支持不区分大小写的查找。

    参数:
        task_type: 调用方提供的任务类型名称。

    返回值:
        去除首尾空白并转换为小写的名称。
    """
    return task_type.strip().lower()


class WorkflowRegistry:
    """按规范化任务类型保存显式注册的固定工作流。"""

    def __init__(self, workflows: dict[str, Workflow]) -> None:
        """校验注册项，并使用规范化键保存。

        参数:
            workflows: 原始任务类型到固定工作流的映射。

        异常:
            ValueError: 多个名称规范化后发生冲突时抛出。
        """
        self._workflows: dict[str, Workflow] = {}
        for raw_name, workflow in workflows.items():
            key = _normalize(raw_name)
            if key in self._workflows:
                raise ValueError(f"Duplicate task type after normalization: {key}")
            self._workflows[key] = workflow

    def contains(self, task_type: str) -> bool:
        """返回规范化后的任务类型是否已注册。

        参数:
            task_type: 待查询的任务类型名称。

        返回值:
            对应工作流已注册时为 True，否则为 False。
        """
        return _normalize(task_type) in self._workflows

    def get(self, task_type: str) -> Workflow:
        """返回已注册工作流，否则抛出 INVALID_TASK_TYPE。

        参数:
            task_type: 待查询的任务类型名称。

        返回值:
            与规范化任务类型对应的工作流。

        异常:
            AppError: 任务类型未注册时抛出。
        """
        key = _normalize(task_type)
        if key not in self._workflows:
            raise AppError(ErrorCode.INVALID_TASK_TYPE, "Unknown task type")
        return self._workflows[key]

    def names(self) -> tuple[str, ...]:
        """返回排序后的规范化任务类型名称。

        返回值:
            按字典序排列的任务类型名称元组。
        """
        return tuple(sorted(self._workflows))
