"""稳定任务事件的 SSE 编码。"""

from agent_app.schemas.events import TaskEvent


def encode_sse(event: TaskEvent) -> str:
    """将 TaskEvent 编码为原始 SSE 文本块。

    参数:
        event: 已附加任务元数据与序号的任务事件。

    返回值:
        包含 id、event 和 JSON data 字段的 SSE 文本块。
    """
    lines = [
        f"event: {event.type.value}",
        f"id: {event.sequence}",
        f"data: {event.model_dump_json()}",
    ]
    return "\n".join(lines) + "\n\n"
