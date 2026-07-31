"""稳定的应用错误及安全的执行异常归一化。"""

from enum import StrEnum
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError


class ErrorCode(StrEnum):
    """跨 HTTP 与流式任务边界公开的错误码。"""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    INVALID_TASK_TYPE = "INVALID_TASK_TYPE"
    INVALID_AGENT_TYPE = "INVALID_AGENT_TYPE"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """仅包含可安全暴露给调用方信息的应用异常。"""

    def __init__(
        self,
        code: ErrorCode,
        public_message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """初始化可安全跨应用边界传播的异常。

        参数:
            code: 稳定的应用错误码。
            public_message: 可直接返回给调用方的安全消息。
            details: 可选的调用方安全错误详情。
        """
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.details = details


def error_http_status(code: ErrorCode) -> int:
    """将稳定错误码转换为对应的 HTTP 响应状态码。

    参数:
        code: 应用边界公开的稳定错误码。

    返回值:
        与错误类别匹配的 HTTP 状态码。
    """
    if code in {
        ErrorCode.VALIDATION_ERROR,
        ErrorCode.INVALID_PARAMETERS,
        ErrorCode.INVALID_TASK_TYPE,
        ErrorCode.INVALID_AGENT_TYPE,
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
    """保留 AppError，并将 OpenAI 瞬时故障映射为 UPSTREAM_UNAVAILABLE。

    参数:
        error: 执行工作流或调用模型时捕获的原始异常。
        fallback_code: 未知异常使用的稳定错误码。
        fallback_message: 未知异常对外使用的安全消息。

    返回值:
        不泄露内部细节且具有稳定错误码的 AppError。
    """
    if isinstance(error, AppError):
        return error
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "OpenAI is temporarily unavailable")
    return AppError(fallback_code, fallback_message)
