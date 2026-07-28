"""应用的安全结构化日志配置。"""

import logging
from collections.abc import Mapping
from typing import Any

import structlog

_REDACTED_NON_JSON = "[REDACTED_NON_JSON]"


def _safe_json_value(value: Any) -> Any:
    """保留 JSON 原生日志元数据，并遮蔽不支持的值。

    参数:
        value: 即将写入结构化日志的任意值。

    返回值:
        可安全进行 JSON 序列化的值或固定遮蔽文本。
    """
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
    """防止任意对象进入 JSON 渲染器。

    参数:
        _logger: structlog 处理器传入但本函数无需使用的日志器。
        _method_name: 当前日志方法名称。
        event_dict: 待渲染的结构化事件字典。

    返回值:
        所有值均已转换为 JSON 安全形式的事件字典。
    """
    return {key: _safe_json_value(value) for key, value in event_dict.items()}


def configure_logging(log_level: str) -> None:
    """配置包含时间戳与严重级别的 JSON 日志。

    参数:
        log_level: Python 日志级别名称。
    """
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
