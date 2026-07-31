"""将 DeepAgentRuntime 封装在项目接口之后的适配器。"""

from typing import Any

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.executors import ExecutionContext


class DeepAgentAdapter:
    """运行受限 Deep Agent，并将输出映射为项目事件。"""

    def __init__(self, runtime: DeepAgentRuntime) -> None:
        """保存每次执行时需要流式调用的运行时。

        参数:
            runtime: 实现项目流式运行协议的受限 Deep Agent。
        """
        self._runtime = runtime

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        """流式执行 Agent、发出映射后的事件并返回最终答案。

        参数:
            context: 当前任务输入、检查点消息、运行配置和事件出口。

        返回值:
            包含最终 answer 字段的结果字典。

        异常:
            AppError: 运行时失败或未产生最终答案时抛出。
        """
        agent_input: dict[str, Any] = {"messages": list(context.messages)}

        answer_parts: list[str] = []
        try:
            async for stream_mode, chunk_data in self._runtime.astream(
                agent_input,
                context.config,
                stream_mode=("messages", "updates"),
            ):
                if stream_mode != "messages":
                    continue
                msg = chunk_data[0] if isinstance(chunk_data, tuple) else chunk_data
                event = map_deep_agent_event(msg)
                if event is not None:
                    context.emit(event)
                    if event.type is not None and "delta" in event.data:
                        answer_parts.append(event.data["delta"])
        except AppError:
            raise
        except Exception as error:
            raise normalize_execution_error(
                error,
                fallback_code=ErrorCode.EXECUTION_FAILED,
                fallback_message="Deep Agent execution failed",
            ) from error

        answer = "".join(answer_parts).strip()
        if not answer:
            raise AppError(
                ErrorCode.EXECUTION_FAILED,
                "Deep Agent returned no answer",
            )
        return {"answer": answer}
