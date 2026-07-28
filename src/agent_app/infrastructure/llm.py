"""OpenAI chat model factory."""

from langchain_openai import ChatOpenAI

from agent_app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a ChatOpenAI instance from settings without import-time construction."""
    return ChatOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
