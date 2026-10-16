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


def test_public_read_download_url_is_unsigned_and_encoded() -> None:
    """public_read 模式：download_url 返回无签名公开链接，中文 key 百分号编码。"""
    client = _FakeBotoClient()
    storage = Boto3Storage(
        client=client,
        bucket="beauty",
        expires_in=300,
        public_read=True,
        public_base="https://s3.public.test/beauty",
    )
    url = storage.download_url("图片/人物/白丝/示例-白丝-高挑.png")
    assert url == (
        "https://s3.public.test/beauty/"
        "%E5%9B%BE%E7%89%87/%E4%BA%BA%E7%89%A9/%E7%99%BD%E4%B8%9D/"
        "%E7%A4%BA%E4%BE%8B-%E7%99%BD%E4%B8%9D-%E9%AB%98%E6%8C%91.png"
    )
    assert "X-Amz" not in url
    # GET 不再走签名 client（上传仍预签名）。
    assert not client.url_calls or client.url_calls[0]["operation"] == "put_object"


def test_public_read_upload_url_stays_presigned() -> None:
    """公开读只影响下载链接：上传仍返回预签名 PUT（桶写入未公开）。"""
    client = _FakeBotoClient(presigned="https://signed.test/put")
    storage = Boto3Storage(
        client=client,
        bucket="beauty",
        expires_in=300,
        public_read=True,
        public_base="https://s3.public.test/beauty",
    )
    assert storage.upload_url("k.png", content_type="image/png") == "https://signed.test/put"
    assert client.url_calls[0]["operation"] == "put_object"


def test_factory_public_read_without_public_domain_falls_back_to_presigned(monkeypatch) -> None:
    """public_read 但未配置 public_domain 时回退预签名（无公开端点可拼）。"""
    from agent_app.config import S3Config
    from agent_app.infrastructure.storage import create_object_storage

    class FakeSession:
        def client(self, service, **options):
            return _FakeBotoClient(presigned="https://signed.test/x")

    monkeypatch.setattr("agent_app.infrastructure.storage.boto3.session.Session", FakeSession)
    storage = create_object_storage(
        S3Config(endpoint_url="http://internal.test:9000", public_read=True)
    )
    assert storage.download_url("k") == "https://signed.test/x"


def test_list_returns_entries_with_metadata() -> None:
    """list 返回含 size 与 last_modified 的 ObjectEntry（自动翻页）。"""
    from datetime import UTC, datetime

    from agent_app.infrastructure.storage import ObjectEntry

    modified = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    class FakePaginator:
        def paginate(self, **kwargs):
            assert kwargs["Prefix"] == "tmp/"
            yield {"Contents": [{"Key": "tmp/a.bin", "Size": 12, "LastModified": modified}]}
            yield {"Contents": [{"Key": "tmp/b.bin", "Size": 3}]}

    class ListingClient:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

    storage = Boto3Storage(client=ListingClient(), bucket="b", expires_in=300)

    entries = storage.list("tmp/")

    assert entries == [
        ObjectEntry(key="tmp/a.bin", size=12, last_modified=modified.isoformat()),
        ObjectEntry(key="tmp/b.bin", size=3, last_modified=None),
    ]


def test_list_maps_client_error_to_upstream_unavailable() -> None:
    class FailingPaginator:
        def paginate(self, **kwargs):
            raise ClientError({"Error": {"Code": "X", "Message": "y"}}, "ListObjectsV2")
            yield  # pragma: no cover - 使其成为生成器函数

    class FailingClient:
        def get_paginator(self, name):
            return FailingPaginator()

    storage = Boto3Storage(client=FailingClient(), bucket="b", expires_in=300)

    with pytest.raises(AppError) as error:
        storage.list("tmp/")

    assert error.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
