"""对象存储封装：上传字节并返回可下载的预签名 URL。

通用能力，不绑定 PDF；任何需要把字节产物托管到 S3 兼容服务并换取下载链接的内部工具
均可复用。生产用 ``Boto3Storage``，测试注入实现 ``ObjectStorage`` 协议的内存替身。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import boto3
from botocore.exceptions import ClientError
from pydantic import SecretStr

from agent_app.errors import AppError, ErrorCode

if TYPE_CHECKING:
    from agent_app.config import S3Config


@runtime_checkable
class ObjectStorage(Protocol):
    """上传字节并生成可下载 URL 的对象存储契约。"""

    def put(self, data: bytes, *, key: str, content_type: str) -> None:
        """上传一段字节到指定 key。"""
        ...

    def download_url(self, key: str) -> str:
        """返回可下载指定 key 的（预签名）URL。"""
        ...


class Boto3Storage:
    """基于 boto3 client 的 ``ObjectStorage`` 实现。

    依赖注入 client 以便单测替换；上传/取 URL 失败统一映射为 ``UPSTREAM_UNAVAILABLE``。
    """

    def __init__(self, *, client: Any, bucket: str | None, expires_in: int) -> None:
        """绑定 boto3 client、bucket 与预签名有效期。

        参数:
            client: 已构造的 boto3 S3 client（测试可注入记录型替身）。
            bucket: 目标 bucket 名；允许为 None 以支持延迟配置。
            expires_in: 预签名 URL 的有效秒数。
        """
        self._client = client
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
            带签名、可在 ``expires_in`` 内下载的 URL。
        """
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=self._expires_in,
            )
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
    client = boto3.session.Session().client(
        "s3",
        endpoint_url=s3.endpoint_url,
        region_name=s3.region,
        aws_access_key_id=_secret(s3.access_key),
        aws_secret_access_key=_secret(s3.secret_key),
    )
    return Boto3Storage(
        client=client,
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
