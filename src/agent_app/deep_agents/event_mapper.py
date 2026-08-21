"""将 deepagents 流式数据块映射为项目标准化的 PendingEvent。"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent_app.schemas.events import EventType, PendingEvent


def map_deep_agent_event(message: Any) -> list[PendingEvent] | None:
    """将流式消息转换为 PendingEvent 列表；无需处理时返回 None。

    messages 流模式下，流式模型产出 AIMessageChunk；非流式模型
    （langgraph 会包装其结果）产出完整 AIMessage，两者均需处理。

    流式工具调用中，中间分片的 ``tool_calls`` 可能出现 ``name`` 为空
    字符串的聚合状态，本函数会过滤这些空名分片，避免同一工具调用
    产生多个带空名的 ``tool.started`` 事件。

    参数:
        message: deepagents 运行时产生的消息块或状态更新对象。

    返回值:
        映射后的标准化待处理事件列表；消息不受支持或无内容时返回 None。
    """
    if isinstance(message, (AIMessage, AIMessageChunk)):
        if message.tool_calls:
            events: list[PendingEvent] = []
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                if not name:
                    continue
                events.append(
                    PendingEvent(
                        type=EventType.TOOL_STARTED,
                        data={"tool_name": name},
                    )
                )
            return events if events else None
        if message.content:
            return [
                PendingEvent(
                    type=EventType.CONTENT_DELTA,
                    data={"delta": message.content},
                )
            ]
        return None

    if isinstance(message, ToolMessage):
        status = getattr(message, "status", "success")
        return [
            PendingEvent(
                type=EventType.TOOL_COMPLETED,
                data={
                    "tool_name": getattr(message, "name", "unknown"),
                    "status": status,
                },
            )
        ]

    return None
