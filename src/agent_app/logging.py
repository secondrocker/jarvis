"""Safe structured logging configuration for the application."""

import logging
from collections.abc import Mapping
from typing import Any

import structlog

_REDACTED_NON_JSON = "[REDACTED_NON_JSON]"


def _safe_json_value(value: Any) -> Any:
    """Keep JSON-native logging metadata while redacting unsupported values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return _REDACTED_NON_JSON
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_json_value(item) for item in value]
    return _REDACTED_NON_JSON


def redact_non_json_values(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Prevent arbitrary objects from reaching the JSON renderer."""
    return {key: _safe_json_value(value) for key, value in event_dict.items()}


def configure_logging(log_level: str) -> None:
    """Configure JSON logs containing timestamps and severity levels."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_log_level,
            redact_non_json_values,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
