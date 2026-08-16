"""将 deepagents 流式数据块映射为项目标准化的 PendingEvent。"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent_app.schemas.events import EventType, PendingEvent


def map_deep_agent_event(message: Any) -> PendingEvent | None:
    """将流式消息转换为 PendingEvent；无需处理时返回 None。

    参数:
        message: deepagents 运行时产生的消息块或状态更新对象。

    返回值:
        映射后的标准化待处理事件；消息不受支持或无内容时返回 None。
    """
    # messages 流模式下，流式模型产出 AIMessageChunk；非流式模型
    # （langgraph 会包装其结果）产出完整 AIMessage，两者均需处理。
    if isinstance(message, (AIMessage, AIMessageChunk)):
        if message.tool_calls:
            tool_name = message.tool_calls[0].get("name", "unknown")
            return PendingEvent(
                type=EventType.TOOL_STARTED,
                data={"tool_name": tool_name},
            )
        if message.content:
            return PendingEvent(
                type=EventType.CONTENT_DELTA,
                data={"delta": message.content},
            )
        return None

    if isinstance(message, ToolMessage):
        status = getattr(message, "status", "success")
        return PendingEvent(
            type=EventType.TOOL_COMPLETED,
            data={
                "tool_name": getattr(message, "name", "unknown"),
                "status": status,
            },
        )

    return None
