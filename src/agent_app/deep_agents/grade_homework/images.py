"""批改作业图片的下载、安全校验与多模态消息装配。

感知层是确定性工作：下载、拒绝内网地址、兜底压缩为 JPEG、拼成
``data:image/jpeg;base64,...``。判断层（判分、错因、评语）留给 agent。
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.messages import AnyMessage, HumanMessage
from PIL import Image

from agent_app.deep_agents.grade_homework.calc import (
    CalcError,
    assert_public_http_url,
)
from agent_app.deep_agents.grade_homework.schemas import GradeInput
from agent_app.errors import AppError, ErrorCode

# message 中多个 URL 的分隔符（空白、逗号、中文逗号、分号）。
_SPLIT_RE = re.compile(r"[\s,，;；]+")

# 送模型的图片 MIME：统一转 JPEG。
_JPEG_PREFIX = "data:image/jpeg;base64,"


def extract_urls(message: str, parameters: dict[str, Any] | None = None) -> list[str]:
    """从 ``parameters``（优先）或 ``message`` 中提取作业图片 URL 列表。

    parameters 里可显式给 ``urls``（列表）或 ``url``（单个）；都没有时把
    message 按空白/逗号切分，并只保留形如 URL 的片段（含 ``://``）——
    这样"第二题错在哪"这类追问会被识别为无新图片（图片已在会话历史里），
    而不是被当成一个 URL 去下载。

    参数:
        message: 任务消息（通常是一个或多个图片 URL）。
        parameters: 调用方附加参数。

    返回值:
        提取到的 URL 列表（可能为空）。
    """
    payload: dict[str, Any] = dict(parameters or {})
    given: Any = payload.get("urls")
    if given is None and payload.get("url"):
        given = [payload["url"]]
    if given is None:
        return [url for url in _SPLIT_RE.split(message.strip()) if _looks_like_url(url)]
    if isinstance(given, str):
        return [given]
    return [str(url) for url in given]


def _looks_like_url(token: str) -> bool:
    """判断片段是否像 URL（含协议分隔符或 http(s) 前缀）。"""
    lowered = token.lower()
    return "://" in token or lowered.startswith(("http:", "https:"))


def validate_urls(urls: Sequence[str]) -> GradeInput:
    """用公开契约校验 URL 列表（数量上限 8，兼容旧 ``url`` 字段）。

    参数:
        urls: 待校验的图片 URL 列表。

    返回值:
        已校验的 :class:`GradeInput`。

    异常:
        ValidationError: 数量不合法时抛出（由调用方包装为公开错误）。
    """
    return GradeInput.model_validate({"urls": list(urls)})


def _fetch_one(
    url: str,
    *,
    max_image_mb: int,
    max_image_edge: int,
    timeout_seconds: float,
) -> str:
    """下载并压缩单张图片，返回 JPEG data URL。"""
    try:
        assert_public_http_url(url)
    except CalcError as error:
        raise AppError(ErrorCode.INVALID_PARAMETERS, str(error)) from error
    try:
        response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.INVALID_PARAMETERS,
            f"failed to download image: {error}",
        ) from error
    data = response.content
    if len(data) > max_image_mb * 1024 * 1024:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "image exceeds size limit")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if max(image.size) > max_image_edge:
            image.thumbnail((max_image_edge, max_image_edge))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    except Exception as error:  # noqa: BLE001 - 统一转为参数错误
        raise AppError(ErrorCode.INVALID_PARAMETERS, f"cannot decode image: {error}") from error
    return _JPEG_PREFIX + base64.b64encode(buffer.getvalue()).decode()


def fetch_image_data_urls(
    urls: Sequence[str],
    *,
    max_image_mb: int = 8,
    max_image_edge: int = 2048,
    timeout_seconds: float = 30.0,
) -> list[str]:
    """逐张下载并压缩作业图片，返回 JPEG data URL 列表。

    参数:
        urls: 可下载的作业图片地址（同一份作业的连续多页）。
        max_image_mb: 单张图片允许的最大体积（MB）。
        max_image_edge: 兜底压缩的长边上限（像素）。
        timeout_seconds: 单张下载超时秒数。

    返回值:
        与 urls 等长的 JPEG data URL 列表。

    异常:
        AppError: 地址不合法、下载失败、超限或无法解码时抛 INVALID_PARAMETERS。
    """
    return [
        _fetch_one(
            url,
            max_image_mb=max_image_mb,
            max_image_edge=max_image_edge,
            timeout_seconds=timeout_seconds,
        )
        for url in urls
    ]


def messages_have_images(messages: Sequence[AnyMessage]) -> bool:
    """判断消息历史里是否已经带有图片（多轮追问时复用同一份作业）。

    参数:
        messages: 当前会话消息列表。

    返回值:
        任意消息含 image_url 内容块时为 True。
    """
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "image_url" for block in content
        ):
            return True
    return False


def attach_images(messages: Sequence[AnyMessage], data_urls: Sequence[str]) -> list[AnyMessage]:
    """把图片内容块附加到当前用户消息（或新建一条用户消息）。

    历史消息按原样保留（多轮会话），只在最后一条用户消息上追加图片；
    最后一条不是用户消息时新建一条，避免图片块落到工具/助手消息里。

    参数:
        messages: 当前会话消息列表。
        data_urls: 待附加的 JPEG data URL 列表。

    返回值:
        可直接交给 agent 运行时的消息列表。
    """
    image_blocks: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url}} for data_url in data_urls
    ]
    history = list(messages)
    if not image_blocks:
        return history
    if history and isinstance(history[-1], HumanMessage):
        last = history[-1]
        text = last.content if isinstance(last.content, str) else ""
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}, *image_blocks]
        history[-1] = last.model_copy(update={"content": blocks})
        return history
    return [
        *history,
        HumanMessage(
            content=[
                {"type": "text", "text": "请批改这份作业的图片。"},
                *image_blocks,
            ]
        ),
    ]
