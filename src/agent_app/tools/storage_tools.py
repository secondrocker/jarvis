"""对象存储能力的 MCP 工具：URL 中转上传与刷新预签名 URL。

底层 ObjectStorage（Boto3Storage）的字节接口不天然映射 MCP 参数，故上传采用 URL 中转：
先下载远端字节再上传 S3，返回 {key, url}。get_download_url 独立暴露，用于刷新过期 URL；
get_upload_url 返回预签名 PUT URL，供调用方直传字节到 S3。
"""

from typing import Any
from uuid import uuid4

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from agent_app.errors import AppError, ErrorCode
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.schemas.storage_tools import (
    StorageDownloadUrlInput,
    StorageUploadFromUrlInput,
    StorageUploadUrlInput,
)

# 下载远端源的最大等待秒数（与 pdf/io.py 一致）。
DOWNLOAD_TIMEOUT = 30.0


def _download(url: str) -> bytes:
    """从 URL 下载字节；失败时抛出安全的 INVALID_PARAMETERS。

    参数:
        url: 已规范化的可下载 URL。

    返回值:
        待上传的原始字节。

    异常:
        AppError: 下载失败时抛出 INVALID_PARAMETERS。
    """
    try:
        response = httpx.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.INVALID_PARAMETERS,
            "Source could not be downloaded",
        ) from error
    return response.content


def register_storage_tools(
    mcp: FastMCP,
    *,
    storage: ObjectStorage,
) -> None:
    """把对象存储 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        storage: 上传字节并换取下载 URL 的对象存储。
    """

    @mcp.tool
    def upload_from_url(
        source_url: str,
        content_type: str,
        key: str | None = None,
    ) -> dict[str, Any]:
        """从远端 URL 下载字节并上传到对象存储，返回可下载 URL。

        source_url 为可下载的远端资源地址；content_type 写入对象元数据；
        key 可选，为对象的完整存储路径（固定使用传入路径，不再追加随机段）；
        缺省时生成随机路径 uploads/<uuid>。

        Args:
            source_url: 可下载的远端资源 URL。
            content_type: 写入对象的 Content-Type。
            key: 对象的完整 S3 key；缺省生成 uploads/<uuid>。

        返回值:
            含 key、url 的字典。
        """
        try:
            payload = StorageUploadFromUrlInput(
                source_url=source_url,
                content_type=content_type,
                key=key,
            )
            data = _download(payload.source_url)
            object_key = payload.key or f"uploads/{uuid4().hex}"
            storage.put(data, key=object_key, content_type=payload.content_type)
            return {"key": object_key, "url": storage.download_url(object_key)}
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def get_download_url(key: str) -> dict[str, Any]:
        """为已存在的对象生成可下载的预签名 URL。

        key 为对象存储 key；返回可在有效期内下载的 URL。

        Args:
            key: 对象存储 key。

        返回值:
            含 key、url 的字典。
        """
        try:
            payload = StorageDownloadUrlInput(key=key)
            return {"key": payload.key, "url": storage.download_url(payload.key)}
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def get_upload_url(
        content_type: str,
        key: str | None = None,
    ) -> dict[str, Any]:
        """生成可直接 PUT 上传字节的预签名 URL，供调用方直传 S3。

        content_type 会写入签名，调用方 PUT 时必须携带匹配的 ``Content-Type`` header。
        key 可选，为对象的完整存储路径（固定使用传入路径，不再追加随机段）；
        缺省时生成随机路径 uploads/<uuid>。

        Args:
            content_type: 签名绑定的 Content-Type。
            key: 对象的完整 S3 key；缺省生成 uploads/<uuid>。

        返回值:
            含 key、url、content_type 的字典。
        """
        try:
            payload = StorageUploadUrlInput(
                content_type=content_type,
                key=key,
            )
            object_key = payload.key or f"uploads/{uuid4().hex}"
            return {
                "key": object_key,
                "url": storage.upload_url(object_key, content_type=payload.content_type),
                "content_type": payload.content_type,
            }
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error
