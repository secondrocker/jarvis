"""Stable application errors and safe execution-error normalization."""

from enum import StrEnum
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError


class ErrorCode(StrEnum):
    """Error codes exposed across HTTP and streamed task boundaries."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    INVALID_TASK_TYPE = "INVALID_TASK_TYPE"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """An application error containing only caller-safe information."""

    def __init__(
        self,
        code: ErrorCode,
        public_message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.details = details


def error_http_status(code: ErrorCode) -> int:
    """Translate stable error codes to their HTTP response status."""
    if code in {
        ErrorCode.VALIDATION_ERROR,
        ErrorCode.INVALID_PARAMETERS,
        ErrorCode.INVALID_TASK_TYPE,
    }:
        return 422
    if code is ErrorCode.UPSTREAM_UNAVAILABLE:
        return 503
    return 500


def normalize_execution_error(
    error: Exception,
    *,
    fallback_code: ErrorCode,
    fallback_message: str,
) -> AppError:
    """Preserve AppError and map OpenAI transient failures to UPSTREAM_UNAVAILABLE."""
    if isinstance(error, AppError):
        return error
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI is temporarily unavailable")
    return AppError(fallback_code, fallback_message)
