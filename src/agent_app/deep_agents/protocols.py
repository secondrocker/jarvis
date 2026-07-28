"""Protocols isolating the deepagents library boundary from project code."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig


@runtime_checkable
class DeepAgentRuntime(Protocol):
    """Contract for a restricted deep agent capable of streaming execution."""

    def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig,
        *,
        stream_mode: tuple[str, ...],
    ) -> AsyncIterator[Any]:
        """Stream message and update chunks from the restricted runtime."""
        ...
