"""Top-level orchestration graph state."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state flowing through the top-level orchestration graph."""

    task_id: str
    thread_id: str
    message: str
    messages: Annotated[list[AnyMessage], add_messages]
    execution_mode: str
    requested_task_type: str | None
    parameters: dict[str, Any]
    selected_mode: str | None
    selected_task_type: str | None
    route_reason: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
