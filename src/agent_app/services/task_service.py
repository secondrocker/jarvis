"""封装已编译编排图的统一任务执行服务。"""

import asyncio
from collections.abc import AsyncIterator, Collection
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from agent_app.errors import AppError, ErrorCode
from agent_app.orchestration.state import AgentState
from agent_app.schemas.events import EventSequencer, EventType, PendingEvent, TaskEvent
from agent_app.schemas.tasks import (
    ExecutionInfo,
    ExecutionMode,
    SelectedMode,
    TaskRequest,
    TaskResponse,
)


class TaskService:
    """为同步与流式执行提供唯一的内部事件源。"""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        registered_task_types: Collection[str],
        task_timeout_seconds: float,
    ) -> None:
        """保存编排图、规范化任务名称和正数超时时间。

        参数:
            graph: 已编译的顶层编排图。
            registered_task_types: 可通过工作流执行的任务类型集合。
            task_timeout_seconds: 单次任务的最大执行秒数。

        异常:
            ValueError: 超时时间不是正数时抛出。
        """
        self._graph = graph
        self._task_types = frozenset(t.strip().lower() for t in registered_task_types)
        if task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive")
        self._timeout = task_timeout_seconds

    def preflight(self, request: TaskRequest) -> None:
        """拒绝显式指定但尚未注册任务类型的工作流请求。

        参数:
            request: 即将执行的任务请求。

        异常:
            AppError: 显式工作流请求指定了未注册的任务类型时抛出。
        """
        if request.execution_mode is ExecutionMode.WORKFLOW:
            task_type = (request.task_type or "").strip().lower()
            if task_type not in self._task_types:
                raise AppError(
                    ErrorCode.INVALID_TASK_TYPE,
                    "Unknown task type",
                )

    async def stream(self, request: TaskRequest) -> AsyncIterator[TaskEvent]:
        """确保流中只产生一个开始事件和一个终态事件。

        参数:
            request: 已校验的任务请求。

        返回值:
            按单调序号异步产生的任务事件流。
        """
        self.preflight(request)
        task_id = uuid4().hex
        thread_id = request.thread_id or uuid4().hex
        sequencer = EventSequencer(task_id, thread_id)

        yield sequencer.next(EventType.TASK_STARTED, {"message": request.message})

        graph_input: AgentState = {
            "task_id": task_id,
            "thread_id": thread_id,
            "message": request.message,
            "execution_mode": request.execution_mode.value,
            "requested_task_type": request.task_type,
            "parameters": request.parameters,
        }
        config = {"configurable": {"thread_id": thread_id}}

        final_values: dict = {}
        terminal: TaskEvent | None = None
        try:
            async with asyncio.timeout(self._timeout):
                async for chunk in self._graph.astream(
                    graph_input,
                    config,
                    stream_mode=("custom", "values"),
                ):
                    if isinstance(chunk, dict) and "pending_event" in chunk:
                        pending = PendingEvent(**chunk["pending_event"])
                        yield sequencer.next(pending.type, pending.data)
                    elif isinstance(chunk, dict):
                        final_values = chunk
        except TimeoutError:
            terminal = sequencer.next(
                EventType.TASK_FAILED,
                {"code": ErrorCode.EXECUTION_FAILED.value, "reason": "task timeout"},
            )
            yield terminal
            return
        except AppError as error:
            terminal = sequencer.next(
                EventType.TASK_FAILED,
                {"code": error.code.value, "reason": error.public_message},
            )
            yield terminal
            return
        except Exception:
            terminal = sequencer.next(
                EventType.TASK_FAILED,
                {"code": ErrorCode.INTERNAL_ERROR.value, "reason": "internal error"},
            )
            yield terminal
            return

        if final_values.get("error"):
            err = final_values["error"]
            terminal = sequencer.next(
                EventType.TASK_FAILED,
                {
                    "code": err.get("code", ErrorCode.EXECUTION_FAILED.value),
                    "reason": err.get("message", "execution failed"),
                },
            )
            yield terminal
            return

        result = final_values.get("result") or {}
        selected_mode = final_values.get("selected_mode") or SelectedMode.DEEP_AGENT.value
        selected_task_type = final_values.get("selected_task_type")
        route_reason = final_values.get("route_reason") or ""
        terminal = sequencer.next(
            EventType.TASK_COMPLETED,
            {
                "selected_mode": selected_mode,
                "task_type": selected_task_type,
                "route_reason": route_reason,
                "result": result,
            },
        )
        yield terminal

    async def invoke(self, request: TaskRequest) -> TaskResponse:
        """消费 stream()，并返回校验后的任务完成结果。

        参数:
            request: 已校验的任务请求。

        返回值:
            从完成事件聚合得到的同步任务响应。

        异常:
            AppError: 事件流以失败终态结束或缺少终态时抛出。
        """
        async for event in self.stream(request):
            if event.type is EventType.TASK_COMPLETED:
                data = event.data
                return TaskResponse(
                    task_id=event.task_id,
                    thread_id=event.thread_id,
                    execution=ExecutionInfo(
                        selected_mode=SelectedMode(data["selected_mode"]),
                        task_type=data.get("task_type"),
                        route_reason=data.get("route_reason", ""),
                    ),
                    result=data["result"],
                )
            if event.type is EventType.TASK_FAILED:
                data = event.data
                raise AppError(
                    ErrorCode(data.get("code", ErrorCode.INTERNAL_ERROR.value)),
                    data.get("reason", "internal error"),
                )
        raise AppError(ErrorCode.INTERNAL_ERROR, "stream ended without a terminal event")
