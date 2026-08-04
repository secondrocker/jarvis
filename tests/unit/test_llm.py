"""聊天模型工厂测试。"""

from agent_app.config import Settings
from agent_app.infrastructure.llm import create_chat_model


def test_create_chat_model_uses_global_default_without_override() -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": "test-key", "model": "default-model"}}
    )

    model = create_chat_model(settings)

    assert model.model_name == "default-model"


def test_create_chat_model_uses_executor_specific_model_override() -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": "test-key", "model": "default-model"}}
    )

    model = create_chat_model(settings, model_name="summary-model")

    assert model.model_name == "summary-model"
