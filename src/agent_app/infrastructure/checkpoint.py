"""Process-local checkpoint factory."""

from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> MemorySaver:
    """Create one process-local MemorySaver for the application lifespan.

    Do NOT cache at module import time; the FastAPI lifespan must call this
    so tests can create isolated apps with their own checkpoint state.
    """
    return MemorySaver()
