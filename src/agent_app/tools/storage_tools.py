"""对象存储能力的 MCP 工具：URL 中转上传与刷新预签名 URL。

底层 ObjectStorage（Boto3Storage）的字节接口不天然映射 MCP 参数，故上传采用 URL 中转：
先下载远端字节再上传 S3，返回 {key, url}。get_download_url 独立暴露，用于刷新过期 URL；
get_upload_url 返回预签名 PUT URL，供调用方直传字节到 S3；delete_object 删除对象；
rename_object 重命名/移动对象路径（S3 语义为复制后删除源对象）；
list_objects 列出指定路径（前缀）下的全部对象及其元数据；
move_objects/delete_objects 提供批量移动与批量删除（尽力而为 + 逐项结果汇总）。
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
    StorageBatchDeleteInput,
    StorageBatchMoveInput,
    StorageDeleteInput,
    StorageDownloadUrlInput,
    StorageListInput,
    StorageRenameInput,
    StorageUploadFromUrlInput,
    StorageUploadUrlInput,
)

# 下载远端源的最大等待秒数（与 pdf/io.py 一致）。
DOWNLOAD_TIMEOUT = 30.0

# 批量删除的默认数量上限；超过时必须显式 force=True 才会执行。
BATCH_DELETE_LIMIT = 100


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
        key 可选，为对象的完整存储路径（去除首部斜杠后固定使用传入路径，
        不再追加随机段）；缺省时生成随机路径 uploads/<uuid4 hex>。

        Args:
            source_url: 可下载的远端资源 URL。
            content_type: 写入对象的 Content-Type。
            key: 对象的完整 S3 key；缺省生成 uploads/<uuid4 hex>。

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
    def delete_object(key: str) -> dict[str, Any]:
        """删除对象存储中的指定对象。

        key 为对象存储 key；删除操作幂等（对象不存在时同样视为成功）。

        Args:
            key: 待删除对象的 S3 key。

        返回值:
            含 key、deleted=True 的字典。
        """
        try:
            payload = StorageDeleteInput(key=key)
            storage.delete(payload.key)
            return {"key": payload.key, "deleted": True}
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def rename_object(source_key: str, target_key: str) -> dict[str, Any]:
        """重命名/移动对象存储中的对象路径。

        把对象从 source_key 移动到 target_key（S3 语义为复制后删除源对象）。
        源对象不存在时复制会失败并以 ToolError 报错。

        Args:
            source_key: 源对象的 S3 key。
            target_key: 目标对象的 S3 key。

        返回值:
            含 source_key、target_key、url（新路径的预签名下载 URL）的字典。
        """
        try:
            payload = StorageRenameInput(source_key=source_key, target_key=target_key)
            storage.rename(payload.source_key, payload.target_key)
            return {
                "source_key": payload.source_key,
                "target_key": payload.target_key,
                "url": storage.download_url(payload.target_key),
            }
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
        key 可选，为对象的完整存储路径（去除首部斜杠后固定使用传入路径，
        不再追加随机段）；缺省时生成随机路径 uploads/<uuid4 hex>。

        Args:
            content_type: 签名绑定的 Content-Type。
            key: 对象的完整 S3 key；缺省生成 uploads/<uuid4 hex>。

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

    @mcp.tool
    def list_objects(path: str = "") -> dict[str, Any]:
        """列出对象存储中指定路径下的全部对象，返回文件名与元数据列表。

        path 为目录语义的存储路径（S3 中即 key 前缀，如 "images/animals/"）；
        自动补齐尾部斜杠，避免误命中同名前缀的对象；缺省或空字符串表示
        列举 bucket 根下的全部对象。

        Args:
            path: 目录语义的存储路径前缀，可选。

        返回值:
            含 path、count、keys（完整对象 key 列表）、files（对象列表，每项含
            key、filename——key 去除 path 前缀后的相对文件名，以及 size 字节数
            与 last_modified 最近修改时间）的字典。
        """
        try:
            payload = StorageListInput(path=path)
            entries = storage.list(payload.path)
            files = [
                {
                    "key": entry.key,
                    "filename": entry.key[len(payload.path) :],
                    "size": entry.size,
                    "last_modified": entry.last_modified,
                }
                for entry in entries
            ]
            return {
                "path": payload.path,
                "count": len(files),
                "keys": [entry.key for entry in entries],
                "files": files,
            }
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def move_objects(
        source_path: str | None = None,
        target_path: str | None = None,
        moves: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """批量移动对象：目录（前缀）级移动或显式 key 映射二选一。

        前缀模式传 source_path + target_path：把 source_path 下全部对象移动到
        target_path 下，保留相对文件名（如 "reports/2025/a.pdf" →
        "archive/2025/a.pdf"）；显式模式传 moves 列表（每项含 source_key 与
        target_key）。两种模式互斥。逐项尽力而为：单项失败不中断，结果中
        汇总 moved/failed 明细；source 与 target 相同的项跳过（skipped）。

        Args:
            source_path: 源目录前缀（前缀模式必填）。
            target_path: 目标目录前缀（前缀模式必填，空字符串表示根目录）。
            moves: 显式 {source_key, target_key} 映射列表（映射模式必填）。

        返回值:
            含 moved（每项含 source_key/target_key/url）、skipped（源目标相同的
            key）、failed（每项含 source_key/error）、moved_count、failed_count
            的字典。
        """
        try:
            payload = StorageBatchMoveInput(
                source_path=source_path,
                target_path=target_path,
                moves=moves,
            )
            if payload.moves is not None:
                pairs = [(item.source_key, item.target_key) for item in payload.moves]
            else:
                # 前缀模式：展开为逐对象映射，保留相对文件名。
                source = payload.source_path or ""
                target = payload.target_path or ""
                pairs = [
                    (entry.key, f"{target}{entry.key[len(source) :]}")
                    for entry in storage.list(source)
                ]

            moved: list[dict[str, Any]] = []
            skipped: list[str] = []
            failed: list[dict[str, str]] = []
            for source_key, target_key in pairs:
                if source_key == target_key:
                    skipped.append(source_key)
                    continue
                try:
                    storage.rename(source_key, target_key)
                    moved.append(
                        {
                            "source_key": source_key,
                            "target_key": target_key,
                            "url": storage.download_url(target_key),
                        }
                    )
                except AppError as error:
                    failed.append({"source_key": source_key, "error": error.public_message})
            return {
                "moved": moved,
                "skipped": skipped,
                "failed": failed,
                "moved_count": len(moved),
                "failed_count": len(failed),
            }
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def delete_objects(
        path: str | None = None,
        keys: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """批量删除对象：路径（前缀）模式或显式 key 列表模式二选一。

        路径模式传 path（目录前缀，删除其下全部对象）；列表模式传 keys。
        安全上限：待删对象超过 100 个时默认拒绝执行，需显式 force=True 确认。
        删除幂等（对象不存在视为成功）；逐项尽力而为，结果汇总 deleted/failed。

        Args:
            path: 目录前缀（路径模式必填，空字符串表示整个 bucket 根）。
            keys: 待删对象 key 列表（列表模式必填）。
            force: 超过 100 个对象时是否强制执行。

        返回值:
            含 deleted（成功删除的 key）、failed（每项含 key/error）、
            deleted_count、failed_count 的字典。
        """
        try:
            payload = StorageBatchDeleteInput(path=path, keys=keys, force=force)
            if payload.keys is not None:
                targets = payload.keys
            else:
                targets = [entry.key for entry in storage.list(payload.path or "")]
            if len(targets) > BATCH_DELETE_LIMIT and not payload.force:
                raise AppError(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Refusing to delete {len(targets)} objects; pass force=True to confirm",
                )

            deleted: list[str] = []
            failed: list[dict[str, str]] = []
            for key in targets:
                try:
                    storage.delete(key)
                    deleted.append(key)
                except AppError as error:
                    failed.append({"key": key, "error": error.public_message})
            return {
                "deleted": deleted,
                "failed": failed,
                "deleted_count": len(deleted),
                "failed_count": len(failed),
            }
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid storage parameters"
            raise ToolError(str(detail)) from error
