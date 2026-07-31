import pytest
from pydantic import ValidationError

from agent_app.config import Settings


def test_settings_require_openai_credentials_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert {item["loc"] for item in error.value.errors()} == {
        ("openai_api_key",),
        ("openai_model",),
    }


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
    assert settings.summary_model is None
    assert settings.solution_planning_model is None


def test_settings_allow_executor_specific_models() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="default-model",
        summary_model="summary-model",
        solution_planning_model="planning-model",
        _env_file=None,
    )

    assert settings.summary_model == "summary-model"
    assert settings.solution_planning_model == "planning-model"
