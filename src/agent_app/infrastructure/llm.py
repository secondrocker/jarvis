"""OpenAI 聊天模型工厂。"""

from typing import Any

from langchain_openai import ChatOpenAI

from agent_app.config import Settings


def create_chat_model(
    settings: Settings,
    *,
    model_name: str | None = None,
) -> ChatOpenAI:
    """根据配置创建 ChatOpenAI 实例，避免在模块导入阶段初始化。

    参数:
        settings: 包含模型、认证、端点、超时和重试配置的应用设置。
        model_name: 当前执行器指定的可选模型；未提供时使用全局默认模型。

    返回值:
        已完成参数配置但尚未发起网络请求的聊天模型实例。
    """
    kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key.get_secret_value(),
        "model": model_name or settings.openai_model,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
