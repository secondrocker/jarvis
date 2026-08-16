"""将 DeepAgentRuntime 封装在项目接口之后的适配器。"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.executors import ExecutionContext


def is_from_nested_graph(message: Any, metadata: Any) -> bool:
    """判断 AI 文本块是否产自子代理（作为 task 工具执行的嵌套图）。

    langgraph 的 checkpoint_ns 最后一段标记产生该块的节点：主代理的
    LLM 输出挂在 ``model:<uuid>`` 下；子代理经 task 工具调度，其 LLM
    中间输出挂在 ``tools:<uuid>`` 下（工具执行节点内的嵌套模型调用）。
    以最后一段节点名区分，与图嵌套深度无关（编排图 execute 节点内
    运行时主代理 ns 也含 ``|``，不得按深度过滤）。

    仅对 AIMessage/AIMessageChunk 判定为子代理输出；ToolMessage
    （含 task 工具的结果回传）一律保留——工具执行节点同样挂在
    ``tools:`` 下，但其结果正是主代理需要的输入。

    参数:
        message: 流式消息块。
        metadata: 流式块附带的元数据字典（可能为空或非字典）。

    返回值:
        块为子代理嵌套图的 AI 文本输出时为 True。
    """
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return False
    if not isinstance(metadata, dict):
        return False
    ns = metadata.get("langgraph_checkpoint_ns", "")
    if not isinstance(ns, str) or not ns:
        return False
    return ns.rsplit("|", 1)[-1].startswith("tools:")


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
            # 注意：多流模式必须传 list。langgraph 的 `_output()` 仅在
            # `isinstance(stream_mode, list)` 时产出 `(mode, payload)` 元组，
            # 传 tuple 会退化为裸 payload，导致按模式分发时解包失败。
            async for stream_mode, chunk_data in self._runtime.astream(
                agent_input,
                context.config,
                stream_mode=["messages", "updates"],
            ):
                if stream_mode != "messages":
                    continue
                msg, metadata = chunk_data
                if is_from_nested_graph(msg, metadata):
                    # 防御：子代理（task 工具调度的嵌套图）的中间输出不进入
                    # answer 与事件流。当前 subgraphs=False 时嵌套块本就不会
                    # 泄露到顶层流，此处只防御 deepagents/langgraph 升级改变
                    # 该语义。
                    continue
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
