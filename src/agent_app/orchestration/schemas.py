"""路由决策数据传输模型。"""

from pydantic import BaseModel, Field

from agent_app.schemas.tasks import SelectedMode


class LLMRouteDecision(BaseModel):
    """路由 LLM 使用的结构化输出模型。"""

    selected_mode: SelectedMode
    task_type: str | None = Field(default=None, max_length=64)
    is_ambiguous: bool
    reason: str = Field(min_length=1, max_length=500)


class RouteDecision(BaseModel):
    """按优先级解析后的最终路由决策。"""

    selected_mode: SelectedMode
    task_type: str | None = Field(default=None, max_length=64)
    reason: str = Field(min_length=1, max_length=500)

    @classmethod
    def deep_agent(cls, reason: str, task_type: str | None = None) -> "RouteDecision":
        """使用可安全公开的原因构建 Deep Agent 决策。

        参数:
            reason: 可向调用方返回的路由原因。
            task_type: 可选的原始任务类型。

        返回值:
            执行模式固定为 Deep Agent 的路由决策。
        """
        return cls(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=task_type,
            reason=reason,
        )

    @classmethod
    def workflow(cls, task_type: str, reason: str) -> "RouteDecision":
        """为已注册的任务类型构建工作流决策。

        参数:
            task_type: 已注册的固定工作流任务类型。
            reason: 可向调用方返回的路由原因。

        返回值:
            执行模式固定为工作流的路由决策。
        """
        return cls(
            selected_mode=SelectedMode.WORKFLOW,
            task_type=task_type,
            reason=reason,
        )
