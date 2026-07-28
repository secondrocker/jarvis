"""Top-level LangGraph orchestration with conditional routing."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.deep_agents.adapter import DeepAgentAdapter
from agent_app.errors import ErrorCode
from agent_app.orchestration.registry import WorkflowRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.orchestration.state import AgentState
from agent_app.schemas.events import EventType, PendingEvent
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest

try:
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover — older langgraph fallback
    get_stream_writer = None  # type: ignore[assignment]


def _emit(event_type: EventType, data: dict[str, Any]) -> None:
    """Emit a custom streaming event if a stream writer is available."""
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
        writer({"pending_event": PendingEvent(type=event_type, data=data).model_dump()})
    except Exception:  # pragma: no cover — streaming is best-effort
        pass


def make_normalize_node() -> Callable[[AgentState], dict[str, Any]]:
    """Return the node that appends the user message to checkpoint history."""

    def normalize_input(state: AgentState) -> dict[str, Any]:
        message = state["message"]
        return {"messages": [HumanMessage(content=message)]}

    return normalize_input


def make_route_node(
    router: TaskRouter,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Return the async node that selects the execution path."""

    async def select_route(state: AgentState) -> dict[str, Any]:
        request = TaskRequest(
            message=state["message"],
            execution_mode=ExecutionMode(state["execution_mode"]),
            task_type=state.get("requested_task_type"),
            thread_id=state.get("thread_id"),
            parameters=state.get("parameters", {}),
        )
        decision = await router.route(request)
        _emit(EventType.ROUTE_SELECTED, {
            "selected_mode": decision.selected_mode.value,
            "task_type": decision.task_type,
            "reason": decision.reason,
        })
        return {
            "selected_mode": decision.selected_mode.value,
            "selected_task_type": decision.task_type,
            "route_reason": decision.reason,
        }

    return select_route


def make_workflow_node(
    registry: WorkflowRegistry,
) -> Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Return the async node that executes a registered fixed workflow."""

    async def run_workflow(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        task_type = state.get("selected_task_type") or ""
        workflow = registry.get(task_type)
        _emit(EventType.NODE_STARTED, {"node": f"workflow.{task_type}"})
        try:
            result = await workflow.ainvoke(
                {"text": state["message"], **_extract_summary_params(state)},
                config,
            )
        except Exception as error:
            return {"error": {"code": ErrorCode.EXECUTION_FAILED.value, "message": str(error)}}
        _emit(EventType.NODE_COMPLETED, {"node": f"workflow.{task_type}"})
        return {"result": result}

    return run_workflow


def make_deep_agent_node(
    deep_agent: DeepAgentAdapter,
) -> Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Return the async node that streams the restricted deep agent."""

    async def run_deep_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        _emit(EventType.NODE_STARTED, {"node": "deep_agent"})
        result = await deep_agent.run(
            message=state["message"],
            messages=list(state.get("messages", [])),
            config=config,
            emit=lambda event: _emit(event.type, event.data),
        )
        _emit(EventType.NODE_COMPLETED, {"node": "deep_agent"})
        return {"result": result}

    return run_deep_agent


def _route_after_selection(state: AgentState) -> str:
    """Conditional edge: choose workflow or deep_agent based on selected_mode."""
    mode = state.get("selected_mode", "")
    if mode == SelectedMode.WORKFLOW.value:
        return "workflow"
    return "deep_agent"


def _extract_summary_params(state: AgentState) -> dict[str, Any]:
    """Pull optional summary language and max_words from parameters."""
    params = state.get("parameters") or {}
    out: dict[str, Any] = {}
    if "language" in params:
        out["language"] = params["language"]
    if "max_words" in params:
        out["max_words"] = params["max_words"]
    return out


def build_orchestration_graph(
    *,
    router: TaskRouter,
    registry: WorkflowRegistry,
    deep_agent: DeepAgentAdapter,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the normalized, routed, checkpointed top-level graph."""
    graph = StateGraph(AgentState)
    graph.add_node("normalize_input", make_normalize_node())
    graph.add_node("select_route", make_route_node(router))
    graph.add_node("workflow", make_workflow_node(registry))
    graph.add_node("deep_agent", make_deep_agent_node(deep_agent))

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "select_route")
    graph.add_conditional_edges(
        "select_route",
        _route_after_selection,
        {"workflow": "workflow", "deep_agent": "deep_agent"},
    )
    graph.add_edge("workflow", END)
    graph.add_edge("deep_agent", END)

    return graph.compile(checkpointer=checkpointer)
