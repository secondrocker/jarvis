"""Boto3Storage 的单元测试（注入 fake boto3 client，不联网）。"""

import pytest
from botocore.exceptions import ClientError

from agent_app.errors import AppError, ErrorCode
from agent_app.infrastructure.storage import Boto3Storage


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


def test_upload_url_returns_presigned_put_with_content_type() -> None:
    client = _FakeBotoClient(presigned="https://signed.test/put")
    storage = Boto3Storage(client=client, bucket="secrets", expires_in=3600)

    url = storage.upload_url("uploads/abc.bin", content_type="image/png")

    assert url == "https://signed.test/put"
    assert client.url_calls == [
        {
            "operation": "put_object",
            "Params": {
                "Bucket": "secrets",
                "Key": "uploads/abc.bin",
                "ContentType": "image/png",
            },
            "ExpiresIn": 3600,
        }
    ]


def test_upload_url_maps_client_error_to_upstream_unavailable() -> None:
    storage = Boto3Storage(client=_FakeBotoClient(fail=True), bucket="secrets", expires_in=3600)

    with pytest.raises(AppError) as error:
        storage.upload_url("k", content_type="image/png")

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE


@pytest.mark.parametrize("operation", ["get", "put"])
def test_presigning_uses_separate_public_client_without_reencoding(operation) -> None:
    """签名 URL 的路径和查询编码必须原样返回；内部 client 只负责对象操作。"""
    internal = _FakeBotoClient()
    expected = "https://public.test/b/a%20b%2B.jpg?X-Amz-Signature=fake%2B%2F%3D&x=a%20b"
    public = _FakeBotoClient(presigned=expected)
    storage = Boto3Storage(client=internal, signing_client=public, bucket="b", expires_in=300)
    if operation == "get":
        actual = storage.download_url("a b+.jpg")
    else:
        actual = storage.upload_url("a b+.jpg", content_type="image/jpeg")
    assert actual == expected
    assert not internal.url_calls
    assert public.url_calls[0]["Params"]["Key"] == "a b+.jpg"
    storage.put(b"image", key="a b+.jpg", content_type="image/jpeg")
    assert len(internal.put_calls) == 1
    assert not public.put_calls


@pytest.mark.parametrize("public_domain", [None, "https://public.test/", "public.test"])
def test_factory_signs_at_public_endpoint_with_sigv4(monkeypatch, public_domain) -> None:
    """公开端点在签名前确定，内部端点保持不变且预签名显式使用 SigV4。"""
    from agent_app.config import S3Config
    from agent_app.infrastructure.storage import create_object_storage

    calls = []

    class FakeSession:
        def client(self, service, **options):
            client = _FakeBotoClient()
            calls.append((service, options, client))
            return client

    monkeypatch.setattr("agent_app.infrastructure.storage.boto3.session.Session", FakeSession)
    storage = create_object_storage(
        S3Config(endpoint_url="http://internal.test:9000", public_domain=public_domain)
    )
    assert calls[0][1]["endpoint_url"] == "http://internal.test:9000"
    assert all(options["config"].signature_version == "s3v4" for _, options, _ in calls)
    assert len(calls) == (2 if public_domain else 1)
    if public_domain:
        assert calls[1][1]["endpoint_url"] == "https://public.test"
    storage.download_url("k")
    assert calls[-1][2].url_calls
