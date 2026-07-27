import pytest
from pydantic import ValidationError

from agent_app.config import Settings


def test_settings_require_openai_credentials_and_model() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_apply_demo_timeout_defaults() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        _env_file=None,
    )

    assert settings.llm_timeout_seconds == 60.0
    assert settings.llm_max_retries == 2
    assert settings.task_timeout_seconds == 300.0
    assert settings.log_level == "INFO"
