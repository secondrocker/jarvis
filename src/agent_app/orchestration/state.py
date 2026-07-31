"""顶层编排图状态。"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """在顶层编排图中流转的共享状态。"""

    task_id: str
    thread_id: str
    message: str
    messages: Annotated[list[AnyMessage], add_messages]
    execution_mode: str
    requested_task_type: str | None
    requested_agent_type: str | None
    parameters: dict[str, Any]
    selected_mode: str | None
    selected_executor_type: str | None
    route_reason: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
