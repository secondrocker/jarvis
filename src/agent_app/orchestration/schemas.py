"""Routing decision DTOs."""

from pydantic import BaseModel, Field

from agent_app.schemas.tasks import SelectedMode


class LLMRouteDecision(BaseModel):
    """Structured output schema consumed by the routing LLM."""

    selected_mode: SelectedMode
    task_type: str | None = Field(default=None, max_length=64)
    is_ambiguous: bool
    reason: str = Field(min_length=1, max_length=500)


class RouteDecision(BaseModel):
    """Final routing decision after precedence resolution."""

    selected_mode: SelectedMode
    task_type: str | None = Field(default=None, max_length=64)
    reason: str = Field(min_length=1, max_length=500)

    @classmethod
    def deep_agent(cls, reason: str, task_type: str | None = None) -> "RouteDecision":
        """Build a Deep Agent decision with a safe reason."""
        return cls(
            selected_mode=SelectedMode.DEEP_AGENT,
            task_type=task_type,
            reason=reason,
        )

    @classmethod
    def workflow(cls, task_type: str, reason: str) -> "RouteDecision":
        """Build a workflow decision for a registered task type."""
        return cls(
            selected_mode=SelectedMode.WORKFLOW,
            task_type=task_type,
            reason=reason,
        )
