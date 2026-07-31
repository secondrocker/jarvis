"""统一执行器路由决策数据传输模型。"""

from pydantic import BaseModel, Field

from agent_app.schemas.tasks import SelectedMode


class LLMRouteDecision(BaseModel):
    """路由 LLM 使用的结构化输出模型。"""

    selected_mode: SelectedMode
    executor_type: str | None = Field(default=None, max_length=64)
    is_ambiguous: bool
    reason: str = Field(min_length=1, max_length=500)


class RouteDecision(BaseModel):
    """按优先级解析后的最终路由决策。"""

    selected_mode: SelectedMode
    executor_type: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)

    @classmethod
    def workflow(cls, executor_type: str, reason: str) -> "RouteDecision":
        """构建固定 workflow 路由决策。"""
        return cls(
            selected_mode=SelectedMode.WORKFLOW,
            executor_type=executor_type,
            reason=reason,
        )

    @classmethod
    def deep_agent(cls, executor_type: str, reason: str) -> "RouteDecision":
        """构建 Deep Agent 路由决策。"""
        return cls(
            selected_mode=SelectedMode.DEEP_AGENT,
            executor_type=executor_type,
            reason=reason,
        )
