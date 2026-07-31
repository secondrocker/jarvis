"""统一 workflow 与 Deep Agent 的顶层 LangGraph 编排。"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.executors import ExecutionContext
from agent_app.orchestration.registry import ExecutorRegistry
from agent_app.orchestration.router import TaskRouter
from agent_app.orchestration.state import AgentState
from agent_app.schemas.events import EventType, PendingEvent
from agent_app.schemas.tasks import ExecutionMode, SelectedMode, TaskRequest

try:
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover — 兼容旧版 langgraph
    get_stream_writer = None  # type: ignore[assignment]


def _emit(event_type: EventType, data: dict[str, Any]) -> None:
    """流写入器可用时发出自定义流式事件。"""
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
        writer({"pending_event": PendingEvent(type=event_type, data=data).model_dump()})
    except Exception:  # pragma: no cover — 流式事件采用尽力而为策略
        pass


def make_normalize_node() -> Callable[[AgentState], dict[str, Any]]:
    """返回将当前用户消息追加到检查点历史的节点。"""

    def normalize_input(state: AgentState) -> dict[str, Any]:
        return {"messages": [HumanMessage(content=state["message"])]}

    return normalize_input


def make_route_node(
    router: TaskRouter,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """返回选择统一执行器的异步节点。"""

    async def select_route(state: AgentState) -> dict[str, Any]:
        request = TaskRequest(
            message=state["message"],
            execution_mode=ExecutionMode(state["execution_mode"]),
            task_type=state.get("requested_task_type"),
            agent_type=state.get("requested_agent_type"),
            thread_id=state.get("thread_id"),
            parameters=state.get("parameters", {}),
        )
        decision = await router.route(request)
        task_type = (
            decision.executor_type if decision.selected_mode is SelectedMode.WORKFLOW else None
        )
        agent_type = (
            decision.executor_type if decision.selected_mode is SelectedMode.DEEP_AGENT else None
        )
        _emit(
            EventType.ROUTE_SELECTED,
            {
                "selected_mode": decision.selected_mode.value,
                "task_type": task_type,
                "agent_type": agent_type,
                "reason": decision.reason,
            },
        )
        return {
            "selected_mode": decision.selected_mode.value,
            "selected_executor_type": decision.executor_type,
            "route_reason": decision.reason,
            "result": None,
            "error": None,
        }

    return select_route


def make_execute_node(
    registry: ExecutorRegistry,
) -> Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """返回按路由结果运行任意已注册执行器的节点。"""

    async def execute(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        mode = SelectedMode(state.get("selected_mode", SelectedMode.DEEP_AGENT.value))
        executor_type = state.get("selected_executor_type") or ""
        definition = registry.get(executor_type, mode=mode)
        node_name = f"{mode.value}.{executor_type}"
        _emit(EventType.NODE_STARTED, {"node": node_name})
        context = ExecutionContext(
            message=state["message"],
            messages=list(state.get("messages", [])),
            parameters=dict(state.get("parameters") or {}),
            config=config,
            emit=lambda event: _emit(event.type, event.data),
        )
        try:
            result = await definition.executor.run(context)
        except AppError as error:
            return {
                "error": {
                    "code": error.code.value,
                    "message": error.public_message,
                }
            }
        except Exception:
            return {
                "error": {
                    "code": ErrorCode.EXECUTION_FAILED.value,
                    "message": "Executor execution failed",
                }
            }
        _emit(EventType.NODE_COMPLETED, {"node": node_name})
        update: dict[str, Any] = {"result": result}
        answer = result.get("answer") if mode is SelectedMode.DEEP_AGENT else None
        if isinstance(answer, str) and answer.strip():
            update["messages"] = [AIMessage(content=answer)]
        return update

    return execute


def build_orchestration_graph(
    *,
    router: TaskRouter,
    registry: ExecutorRegistry,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """编译规范化、路由、统一执行和检查点组成的顶层图。"""
    graph = StateGraph(AgentState)
    graph.add_node("normalize_input", make_normalize_node())
    graph.add_node("select_route", make_route_node(router))
    graph.add_node("execute", make_execute_node(registry))

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "select_route")
    graph.add_edge("select_route", "execute")
    graph.add_edge("execute", END)

    return graph.compile(checkpointer=checkpointer)
