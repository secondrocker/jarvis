import pytest
from pydantic import ValidationError

from agent_app.config import Settings


def test_settings_require_openai_credentials_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError) as error:
        Settings.model_validate({"openai": {}})

    assert {item["loc"] for item in error.value.errors()} == {
        ("openai", "api_key"),
        ("openai", "model"),
    }


def test_settings_apply_demo_timeout_defaults() -> None:
    settings = Settings.model_validate({"openai": {"api_key": "test-key", "model": "test-model"}})

    assert settings.openai.timeout_seconds == 60.0
    assert settings.openai.max_retries == 2
    assert settings.task.timeout_seconds == 300.0
    assert settings.log.level == "INFO"
    assert settings.openai.summary_model is None
    assert settings.openai.solution_planning_model is None


def test_settings_allow_executor_specific_models() -> None:
    settings = Settings.model_validate(
        {
            "openai": {
                "api_key": "test-key",
                "model": "default-model",
                "summary_model": "summary-model",
                "solution_planning_model": "planning-model",
            }
        }
    )

    assert settings.openai.summary_model == "summary-model"
    assert settings.openai.solution_planning_model == "planning-model"
