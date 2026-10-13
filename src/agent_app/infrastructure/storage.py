"""对象存储封装：上传字节并返回可下载的预签名 URL。

通用能力，不绑定 PDF；任何需要把字节产物托管到 S3 兼容服务并换取下载链接的内部工具
均可复用。生产用 ``Boto3Storage``，测试注入实现 ``ObjectStorage`` 协议的内存替身。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import SecretStr

from agent_app.errors import AppError, ErrorCode

if TYPE_CHECKING:
    from agent_app.config import S3Config


@runtime_checkable
class ObjectStorage(Protocol):
    """上传字节并生成可下载/上传 URL 的对象存储契约。"""

    def put(self, data: bytes, *, key: str, content_type: str) -> None:
        """上传一段字节到指定 key。"""
        ...

    def download_url(self, key: str) -> str:
        """返回可下载指定 key 的（预签名）URL。"""
        ...

    def upload_url(self, key: str, *, content_type: str) -> str:
        """返回可直接 PUT 上传字节到指定 key 的（预签名）URL。"""
        ...

    def delete(self, key: str) -> None:
        """删除指定 key 的对象。"""
        ...

    def rename(self, source_key: str, target_key: str) -> None:
        """把对象从 source_key 移动/重命名到 target_key。"""
        ...

    def list(self, prefix: str) -> list[str]:
        """列出指定前缀下的全部对象 key。"""
        ...


class Boto3Storage:
    """基于 boto3 client 的 ``ObjectStorage`` 实现。

    依赖注入 client 以便单测替换；上传/取 URL 失败统一映射为 ``UPSTREAM_UNAVAILABLE``。
    """

    def __init__(
        self,
        *,
        client: Any,
        bucket: str | None,
        expires_in: int,
        signing_client: Any | None = None,
    ) -> None:
        """绑定内部操作 client 与公开端点签名 client，签名后不再修改 URL。"""
        self._client = client
        self._signing_client = signing_client if signing_client is not None else client
        self._bucket = bucket
        self._expires_in = expires_in

    def put(self, data: bytes, *, key: str, content_type: str) -> None:
        """上传字节到 ``bucket/key``，失败映射为安全的 ``UPSTREAM_UNAVAILABLE``。

        参数:
            data: 待上传的原始字节。
            key: 对象 key。
            content_type: 写入对象的 Content-Type。
        """
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error

    def download_url(self, key: str) -> str:
        """生成 ``bucket/key`` 的预签名 GET URL。

        参数:
            key: 对象 key。

        返回值:
            带签名、可在 ``expires_in`` 内下载的 URL（直接按访问端点签名）。
        """
        try:
            url = self._signing_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=self._expires_in,
            )
            return url
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error

    def delete(self, key: str) -> None:
        """删除 ``bucket/key`` 处的对象，失败映射为安全的 ``UPSTREAM_UNAVAILABLE``。

        对象不存在时 S3 的 delete_object 也返回成功（幂等），与生产语义一致。

        参数:
            key: 对象 key。
        """
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error

    def rename(self, source_key: str, target_key: str) -> None:
        """把对象从 ``source_key`` 重命名/移动到 ``target_key``。

        S3 没有原生 rename，实现为 copy_object 到目标 key 后删除源对象；
        复制或删除失败均映射为安全的 ``UPSTREAM_UNAVAILABLE``。

        参数:
            source_key: 源对象 key。
            target_key: 目标对象 key。
        """
        try:
            self._client.copy_object(
                Bucket=self._bucket,
                Key=target_key,
                CopySource={"Bucket": self._bucket, "Key": source_key},
            )
            self._client.delete_object(Bucket=self._bucket, Key=source_key)
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error

    def list(self, prefix: str) -> list[str]:
        """列出 ``bucket`` 中前缀匹配的全部对象 key（自动翻页）。

        参数:
            prefix: 对象 key 前缀（S3 无真实目录，目录语义即前缀匹配）。

        返回值:
            前缀下的全部对象 key 列表（S3 按字典序返回）。

        异常:
            AppError: 列举失败时抛出 UPSTREAM_UNAVAILABLE。
        """
        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    keys.append(item["Key"])
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error
        return keys

    def upload_url(self, key: str, *, content_type: str) -> str:
        """生成 ``bucket/key`` 的预签名 PUT URL，绑定指定 Content-Type。

        返回的 URL 要求 PUT 请求携带匹配的 ``Content-Type`` header，否则签名校验失败。

        参数:
            key: 对象 key。
            content_type: 签名绑定的 Content-Type。

        返回值:
            带签名、可在 ``expires_in`` 内执行 PUT 上传的 URL（直接按访问端点签名）。
        """
        try:
            url = self._signing_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self._bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=self._expires_in,
            )
            return url
        except ClientError as error:
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "Object storage is temporarily unavailable",
            ) from error


def create_object_storage(s3: S3Config) -> ObjectStorage:
    """从 S3 配置构造 ``Boto3Storage``。

    凭证/endpoint 缺失时传 ``None``：boto3 构造 client 不联网、不抛，真正上传/取 URL 才需要
    有效凭证——这让非 PDF 路径（摘要 workflow、contract 测试）在无 S3 配置时也能装配。

    参数:
        s3: S3 对象存储配置。

    返回值:
        基于 s3 配置的 ``Boto3Storage``。
    """
    session = boto3.session.Session()
    options = {
        "region_name": s3.region,
        "aws_access_key_id": _secret(s3.access_key),
        "aws_secret_access_key": _secret(s3.secret_key),
        # 显式固定 SigV4，避免 boto3 的预签名回退改用旧版签名。
        "config": Config(signature_version="s3v4"),
    }
    client = session.client("s3", endpoint_url=s3.endpoint_url, **options)
    signing_client = client
    if s3.public_domain:
        # 公开域名本身是签名端点，协议和 Host 都必须在签名前确定。
        endpoint = s3.public_domain.rstrip("/")
        if "://" not in endpoint:
            endpoint = f"https://{endpoint}"
        signing_client = session.client("s3", endpoint_url=endpoint, **options)
    return Boto3Storage(
        client=client,
        signing_client=signing_client,
        bucket=s3.bucket,
        expires_in=s3.url_expires_seconds,
    )


def _secret(value: SecretStr | None) -> str | None:
    """把可选 SecretStr 规范化为 boto3 可接受的显式凭证或 None。

    参数:
        value: 可选的密钥配置。

    返回值:
        非空密钥字符串或 None。
    """
    if value is None:
        return None
    raw = value.get_secret_value()
    return raw or None
