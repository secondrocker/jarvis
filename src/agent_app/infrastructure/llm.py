"""OpenAI chat model factory."""

from typing import Any

from langchain_openai import ChatOpenAI

from agent_app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a ChatOpenAI instance from settings without import-time construction."""
    kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key.get_secret_value(),
        "model": settings.openai_model,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
