"""Adapter wrapping a DeepAgentRuntime behind a project-facing interface."""

from collections.abc import Callable
from typing import Any

from agent_app.deep_agents.event_mapper import map_deep_agent_event
from agent_app.deep_agents.protocols import DeepAgentRuntime
from agent_app.errors import AppError, ErrorCode, normalize_execution_error
from agent_app.schemas.events import PendingEvent

EmitFn = Callable[[PendingEvent], None]


class DeepAgentAdapter:
    """Run a restricted deep agent and map its output to project events."""

    def __init__(self, runtime: DeepAgentRuntime) -> None:
        """Store the runtime that will be streamed on each run."""
        self._runtime = runtime

    async def run(
        self,
        *,
        message: str,
        messages: list[Any],
        config: dict[str, Any],
        emit: EmitFn,
    ) -> dict[str, str]:
        """Stream the agent, emit mapped events, and return the final answer."""
        from langchain_core.messages import HumanMessage

        full_messages = list(messages) + [HumanMessage(content=message)]
        agent_input: dict[str, Any] = {"messages": full_messages}

        answer_parts: list[str] = []
        try:
            async for stream_mode, chunk_data in self._runtime.astream(
                agent_input,
                config,
                stream_mode=("messages", "updates"),
            ):
                if stream_mode != "messages":
                    continue
                msg = chunk_data[0] if isinstance(chunk_data, tuple) else chunk_data
                event = map_deep_agent_event(msg)
                if event is not None:
                    emit(event)
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
