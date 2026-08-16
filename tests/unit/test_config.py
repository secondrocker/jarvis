import pytest
from pydantic import SecretStr, ValidationError

from agent_app.config import Settings


def test_settings_require_openai_credentials_and_model() -> None:
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
    assert settings.openai.info_price_model is None


def test_settings_allow_executor_specific_models() -> None:
    settings = Settings.model_validate(
        {
            "openai": {
                "api_key": "test-key",
                "model": "default-model",
                "summary_model": "summary-model",
                "solution_planning_model": "planning-model",
                "info_price_model": "info-price-model",
            }
        }
    )

    assert settings.openai.summary_model == "summary-model"
    assert settings.openai.solution_planning_model == "planning-model"
    assert settings.openai.info_price_model == "info-price-model"


def test_web_gateway_defaults_to_disabled_when_section_missing() -> None:
    settings = Settings.model_validate({"openai": {"api_key": "test-key", "model": "test-model"}})

    assert settings.web_gateway.base_url is None
    assert settings.web_gateway.api_token is None
    assert settings.web_gateway.search_timeout_seconds == 15.0
    assert settings.web_gateway.fetch_timeout_seconds == 40.0


def test_web_gateway_parses_full_section_and_normalizes_trailing_slash() -> None:
    settings = Settings.model_validate(
        {
            "openai": {"api_key": "test-key", "model": "test-model"},
            "web_gateway": {
                "base_url": " https://surf.leegoo.ltd/ ",
                "api_token": "gateway-token",
                "search_timeout_seconds": 20,
                "fetch_timeout_seconds": 45,
            },
        }
    )

    assert settings.web_gateway.base_url == "https://surf.leegoo.ltd"
    assert settings.web_gateway.api_token == SecretStr("gateway-token")
    assert settings.web_gateway.search_timeout_seconds == 20.0
    assert settings.web_gateway.fetch_timeout_seconds == 45.0


@pytest.mark.parametrize("partial", [{"base_url": "https://surf.leegoo.ltd"}, {"api_token": "t"}])
def test_web_gateway_rejects_half_configured_section(partial: dict) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "openai": {"api_key": "test-key", "model": "test-model"},
                "web_gateway": partial,
            }
        )


def test_web_gateway_rejects_non_positive_timeouts() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "openai": {"api_key": "test-key", "model": "test-model"},
                "web_gateway": {
                    "base_url": "https://surf.leegoo.ltd",
                    "api_token": "t",
                    "search_timeout_seconds": 0,
                },
            }
        )
