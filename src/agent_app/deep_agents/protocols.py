"""用于隔离 deepagents 库与项目代码边界的协议。"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig


@runtime_checkable
class DeepAgentRuntime(Protocol):
    """支持流式执行的受限 Deep Agent 契约。"""

    def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig,
        *,
        stream_mode: tuple[str, ...],
    ) -> AsyncIterator[Any]:
        """从受限运行时流式返回消息块和状态更新块。

        参数:
            input: 发送给运行时的消息和上下文数据。
            config: 当前 LangGraph 运行配置。
            stream_mode: 运行时需要产生的流式数据类别。

        返回值:
            异步迭代的运行时消息块或状态更新块。
        """
        ...
