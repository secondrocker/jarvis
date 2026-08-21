"""将 DeepAgentRuntime 封装在项目接口之后的适配器。"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.orchestration.executors import ExecutionContext
from agent_app.schemas.events import EventType


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


def _is_nested_chunk(namespace: tuple[str, ...]) -> bool:
    """subgraphs=True 时，非空 namespace 表示消息来自子图（task 工具内）。"""
    return bool(namespace)


def _message_has_tool_activity(message: Any) -> bool:
    """消息包含工具调用或工具结果（放行子代理工具事件但不放行文本）。"""
    if isinstance(message, (AIMessage, AIMessageChunk)):
        return bool(message.tool_calls)
    return type(message).__name__ == "ToolMessage"


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
        seen_tool_call_ids: set[str] = set()
        try:
            # 注意：多流模式必须传 list。langgraph 的 `_output()` 仅在
            # `isinstance(stream_mode, list)` 时产出 `(mode, payload)` 元组，
            # 传 tuple 会退化为裸 payload，导致按模式分发时解包崩溃。
            # subgraphs=True 把嵌套子图（子代理）的消息也拉到顶层流，
            # 我们才能把子代理内部的工具调用事件透传给客户端。
            async for item in self._runtime.astream(
                agent_input,
                context.config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                namespace, stream_mode, chunk_data = self._unpack_stream_item(item)
                if stream_mode != "messages":
                    continue
                msg, metadata = chunk_data
                nested = _is_nested_chunk(namespace)
                if nested and not _message_has_tool_activity(msg):
                    # 子代理中间文本不进入 answer 与事件流。
                    continue
                if not nested and is_from_nested_graph(msg, metadata):
                    # 防御：当 subgraphs 语义回退到旧形态时，仍用 ns 过滤子代理文本。
                    continue

                events = map_deep_agent_event(msg)
                if events is None:
                    continue
                for event in events:
                    if nested and event.type is EventType.CONTENT_DELTA:
                        continue
                    if nested and event.type in (
                        EventType.TOOL_STARTED,
                        EventType.TOOL_COMPLETED,
                    ):
                        event.data["source"] = "subagent"
                    if event.type is EventType.TOOL_STARTED:
                        tool_call_id = self._extract_tool_call_id(msg)
                        if tool_call_id and tool_call_id in seen_tool_call_ids:
                            continue
                        if tool_call_id:
                            seen_tool_call_ids.add(tool_call_id)
                    context.emit(event)
                    if not nested and event.type is not None and "delta" in event.data:
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

    @staticmethod
    def _unpack_stream_item(item: Any) -> tuple[tuple[str, ...], str, Any]:
        """兼容 subgraphs=False（二元组）与 True（三元组）的 chunk 形态。

        返回值:
            (namespace, stream_mode, payload)。subgraphs=False 时 namespace 为空元组。
        """
        if isinstance(item, tuple) and len(item) == 3:
            namespace, stream_mode, chunk_data = item
            return (
                tuple(namespace) if isinstance(namespace, tuple) else (),
                stream_mode,
                chunk_data,
            )
        if isinstance(item, tuple) and len(item) == 2:
            stream_mode, chunk_data = item
            return (), stream_mode, chunk_data
        raise ValueError(f"Unexpected stream item shape: {item!r}")

    @staticmethod
    def _extract_tool_call_id(message: Any) -> str | None:
        """从消息中取出 tool_call id，用于对同一工具调用去重。"""
        if isinstance(message, (AIMessage, AIMessageChunk)) and message.tool_calls:
            return message.tool_calls[0].get("id")
        return None
