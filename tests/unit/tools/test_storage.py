"""Boto3Storage 的单元测试（注入 fake boto3 client，不联网）。"""

import pytest
from botocore.exceptions import ClientError

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.storage import Boto3Storage


class _FakeBotoClient:
    """记录 put_object / generate_presigned_url 调用的 boto3 client 替身。"""

    def __init__(
        self,
        *,
        presigned: str = "https://presigned.test/key",
        fail: bool = False,
    ) -> None:
        self.put_calls: list[dict] = []
        self.url_calls: list[dict] = []
        self._presigned = presigned
        self._fail = fail

    def put_object(self, **kwargs) -> None:
        """记录上传参数，或在 fail 模式下抛 ClientError。"""
        if self._fail:
            raise ClientError({"Error": {"Code": "X", "Message": "y"}}, "PutObject")
        self.put_calls.append(kwargs)

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int) -> str:
        """记录预签名参数并返回预设 URL，或在 fail 模式下抛 ClientError。"""
        if self._fail:
            raise ClientError({"Error": {"Code": "X", "Message": "y"}}, "GeneratePresignedUrl")
        self.url_calls.append({"operation": operation, "Params": Params, "ExpiresIn": ExpiresIn})
        return self._presigned


def test_put_forwards_bucket_key_body_content_type() -> None:
    client = _FakeBotoClient()
    storage = Boto3Storage(client=client, bucket="secrets", expires_in=3600)

    storage.put(b"bytes", key="pdf_images/abc/page_1.png", content_type="image/png")

    assert client.put_calls == [
        {
            "Bucket": "secrets",
            "Key": "pdf_images/abc/page_1.png",
            "Body": b"bytes",
            "ContentType": "image/png",
        }
    ]


def test_download_url_returns_presigned_with_expires_in() -> None:
    client = _FakeBotoClient(presigned="https://signed.test/x")
    storage = Boto3Storage(client=client, bucket="secrets", expires_in=7200)

    url = storage.download_url("pdf_images/abc/page_1.png")

    assert url == "https://signed.test/x"
    assert client.url_calls == [
        {
            "operation": "get_object",
            "Params": {"Bucket": "secrets", "Key": "pdf_images/abc/page_1.png"},
            "ExpiresIn": 7200,
        }
    ]


def test_put_maps_client_error_to_upstream_unavailable() -> None:
    storage = Boto3Storage(client=_FakeBotoClient(fail=True), bucket="secrets", expires_in=3600)

    with pytest.raises(AppError) as error:
        storage.put(b"bytes", key="k", content_type="image/png")

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE


def test_download_url_maps_client_error_to_upstream_unavailable() -> None:
    storage = Boto3Storage(client=_FakeBotoClient(fail=True), bucket="secrets", expires_in=3600)

    with pytest.raises(AppError) as error:
        storage.download_url("k")

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
