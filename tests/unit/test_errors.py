import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from agent_app.errors import (
    AppError,
    ErrorCode,
    error_http_status,
    normalize_execution_error,
)


def test_error_codes_have_stable_http_mapping() -> None:
    error = AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI unavailable")

    assert error_http_status(error.code) == 503
    assert error.public_message == "OpenAI unavailable"


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        (ErrorCode.VALIDATION_ERROR, 422),
        (ErrorCode.INVALID_PARAMETERS, 422),
        (ErrorCode.INVALID_TASK_TYPE, 422),
        (ErrorCode.INVALID_AGENT_TYPE, 422),
        (ErrorCode.UPSTREAM_UNAVAILABLE, 503),
        (ErrorCode.EXECUTION_FAILED, 500),
        (ErrorCode.INTERNAL_ERROR, 500),
    ],
)
def test_every_error_code_has_a_stable_http_status(code: ErrorCode, expected_status: int) -> None:
    assert error_http_status(code) == expected_status


def test_normalize_execution_error_preserves_existing_app_error() -> None:
    original = AppError(
        ErrorCode.INVALID_PARAMETERS,
        "invalid summary options",
        {"field": "length"},
    )

    assert (
        normalize_execution_error(
            original,
            fallback_code=ErrorCode.INTERNAL_ERROR,
            fallback_message="internal failure",
        )
        is original
    )


@pytest.mark.parametrize(
    "upstream_error",
    [
        APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses")),
        APIConnectionError(
            message="secret upstream connection message",
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
        RateLimitError(
            "secret upstream rate-limit message",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            ),
            body=None,
        ),
    ],
)
def test_normalize_execution_error_sanitizes_openai_transient_failures(
    upstream_error: Exception,
) -> None:
    normalized = normalize_execution_error(
        upstream_error,
        fallback_code=ErrorCode.EXECUTION_FAILED,
        fallback_message="execution failed",
    )

    assert normalized.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert normalized.public_message == "OpenAI is temporarily unavailable"
    assert normalized.details is None
    assert str(upstream_error) not in normalized.public_message


def test_normalize_execution_error_uses_the_safe_fallback_for_unknown_errors() -> None:
    normalized = normalize_execution_error(
        RuntimeError("raw exception including a secret"),
        fallback_code=ErrorCode.INTERNAL_ERROR,
        fallback_message="The task could not be completed",
    )

    assert normalized.code is ErrorCode.INTERNAL_ERROR
    assert normalized.public_message == "The task could not be completed"
    assert normalized.details is None
