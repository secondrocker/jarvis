"""支持条件路由的顶层 LangGraph 编排图。"""

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
except ImportError:  # pragma: no cover — 兼容旧版 langgraph
    get_stream_writer = None  # type: ignore[assignment]


def _emit(event_type: EventType, data: dict[str, Any]) -> None:
    """流写入器可用时发出自定义流式事件。

    参数:
        event_type: 待发出的稳定事件类型。
        data: 事件携带的业务数据。
    """
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
        writer({"pending_event": PendingEvent(type=event_type, data=data).model_dump()})
    except Exception:  # pragma: no cover — 流式事件采用尽力而为策略
        pass


def make_normalize_node() -> Callable[[AgentState], dict[str, Any]]:
    """返回将用户消息追加到检查点历史记录的节点。

    返回值:
        接收编排状态并返回新增消息的同步节点。
    """

    def normalize_input(state: AgentState) -> dict[str, Any]:
        """把当前用户消息转换为可累积的 LangChain 消息。

        参数:
            state: 包含当前用户消息的编排状态。

        返回值:
            需要追加到检查点历史记录的消息更新。
        """
        message = state["message"]
        return {"messages": [HumanMessage(content=message)]}

    return normalize_input


def make_route_node(
    router: TaskRouter,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """返回选择执行路径的异步节点。

    参数:
        router: 负责按优先级选择执行路径的任务路由器。

    返回值:
        接收编排状态并返回路由字段的异步节点。
    """

    async def select_route(state: AgentState) -> dict[str, Any]:
        """选择执行路径并发出路由事件。

        参数:
            state: 已完成输入规范化的编排状态。

        返回值:
            选定模式、任务类型和路由原因组成的状态更新。
        """
        request = TaskRequest(
            message=state["message"],
            execution_mode=ExecutionMode(state["execution_mode"]),
            task_type=state.get("requested_task_type"),
            thread_id=state.get("thread_id"),
            parameters=state.get("parameters", {}),
        )
        decision = await router.route(request)
        _emit(
            EventType.ROUTE_SELECTED,
            {
                "selected_mode": decision.selected_mode.value,
                "task_type": decision.task_type,
                "reason": decision.reason,
            },
        )
        return {
            "selected_mode": decision.selected_mode.value,
            "selected_task_type": decision.task_type,
            "route_reason": decision.reason,
        }

    return select_route


def make_workflow_node(
    registry: WorkflowRegistry,
) -> Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """返回执行已注册固定工作流的异步节点。

    参数:
        registry: 用于取得目标固定工作流的注册表。

    返回值:
        接收编排状态与运行配置并返回工作流结果的异步节点。
    """

    async def run_workflow(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """执行选定的固定工作流并清理其返回状态。

        参数:
            state: 包含已选任务类型的编排状态。
            config: 传递给固定工作流的 LangGraph 运行配置。

        返回值:
            仅包含公开结果或标准化错误的状态更新。
        """
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
        # 从子图的完整状态中提取结构化结果，避免向外暴露内部状态字段。
        clean_result = result.get("result", result) if isinstance(result, dict) else {}
        return {"result": clean_result}

    return run_workflow


def make_deep_agent_node(
    deep_agent: DeepAgentAdapter,
) -> Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """返回流式执行受限 Deep Agent 的异步节点。

    参数:
        deep_agent: 隔离第三方运行时的项目适配器。

    返回值:
        接收编排状态与运行配置并返回 Agent 结果的异步节点。
    """

    async def run_deep_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """执行受限 Deep Agent 并转发标准化流式事件。

        参数:
            state: 包含当前消息与历史消息的编排状态。
            config: 传递给 Agent 运行时的 LangGraph 配置。

        返回值:
            包含 Agent 最终答案的状态更新。
        """
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
    """条件边：根据 selected_mode 选择 workflow 或 deep_agent。

    参数:
        state: 已包含路由决策的编排状态。

    返回值:
        下一节点名称 ``workflow`` 或 ``deep_agent``。
    """
    mode = state.get("selected_mode", "")
    if mode == SelectedMode.WORKFLOW.value:
        return "workflow"
    return "deep_agent"


def _extract_summary_params(state: AgentState) -> dict[str, Any]:
    """从 parameters 中提取可选的摘要语言和 max_words。

    参数:
        state: 包含原始任务参数的编排状态。

    返回值:
        仅包含摘要工作流支持字段的参数字典。
    """
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
    """编译具备输入规范化、路由和检查点能力的顶层图。

    参数:
        router: 任务执行路径路由器。
        registry: 固定工作流注册表。
        deep_agent: 受限 Deep Agent 适配器。
        checkpointer: 保存多轮会话状态的检查点存储。

    返回值:
        已连接条件边并绑定检查点的顶层编排图。
    """
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
