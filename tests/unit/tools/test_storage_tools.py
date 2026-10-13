"""对象存储 MCP 工具的进程内测试（FastMCP Client，不起 HTTP 端口、不联网）。"""

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError

from agent_app.errors import AppError, ErrorCode
from agent_app.tools import build_mcp_server, storage_tools


class _FakeResponse:
    """httpx.get 的最小替身，返回预设字节。"""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def fake_storage():
    """可断言 delete/rename 调用记录的 fake 存储。"""
    return FakeObjectStorage()


@pytest.fixture
def mcp_server(test_settings, fake_storage):
    """注入内存 fake 存储的聚合 MCP 服务实例。"""
    return build_mcp_server(settings=test_settings, storage=fake_storage)


@pytest.fixture
def patch_download(monkeypatch):
    """patch storage_tools.httpx.get 返回固定字节，返回可控的下载替身。"""
    captured: dict = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(b"downloaded-bytes")

    monkeypatch.setattr(storage_tools.httpx, "get", fake_get)
    return captured


@pytest.mark.asyncio
async def test_upload_from_url_returns_key_and_url(mcp_server, patch_download) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "upload_from_url",
            {"source_url": "https://src.test/file", "content_type": "image/png"},
        )

    data = result.data
    assert data["key"].startswith("uploads/")
    assert data["url"].startswith("https://fake-s3.test/uploads/")
    assert patch_download["url"] == "https://src.test/file"


@pytest.mark.asyncio
async def test_upload_from_url_uses_exact_key(mcp_server, patch_download) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "upload_from_url",
            {
                "source_url": "https://src.test/file",
                "content_type": "application/pdf",
                "key": "/reports/report.pdf",
            },
        )

    assert result.data["key"] == "reports/report.pdf"
    assert result.data["url"].startswith("https://fake-s3.test/reports/report.pdf")


@pytest.mark.asyncio
async def test_get_download_url_returns_url(mcp_server) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_download_url",
            {"key": "uploads/abc.bin"},
        )

    assert result.data == {"key": "uploads/abc.bin", "url": "https://fake-s3.test/uploads/abc.bin"}


@pytest.mark.asyncio
async def test_upload_from_url_requires_url(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("upload_from_url", {"content_type": "image/png"})


@pytest.mark.asyncio
async def test_get_download_url_requires_key(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_download_url", {"key": "  "})


@pytest.mark.asyncio
async def test_get_upload_url_returns_presigned_put(mcp_server) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_upload_url",
            {"content_type": "image/png"},
        )

    data = result.data
    assert data["key"].startswith("uploads/")
    assert data["url"].startswith("https://fake-s3.test/uploads/")
    assert data["content_type"] == "image/png"


@pytest.mark.asyncio
async def test_get_upload_url_uses_exact_key(mcp_server) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_upload_url",
            {"content_type": "application/pdf", "key": "reports/2026/report.pdf"},
        )

    assert result.data["key"] == "reports/2026/report.pdf"
    assert result.data["url"].startswith("https://fake-s3.test/reports/2026/report.pdf")
    assert result.data["content_type"] == "application/pdf"


@pytest.mark.asyncio
async def test_get_upload_url_requires_content_type(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_upload_url", {"content_type": "  "})


@pytest.mark.asyncio
async def test_delete_object_deletes_key(mcp_server, fake_storage) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "delete_object",
            {"key": "/uploads/old.bin"},
        )

    assert result.data == {"key": "uploads/old.bin", "deleted": True}
    assert fake_storage.deletes == ["uploads/old.bin"]


@pytest.mark.asyncio
async def test_delete_object_requires_key(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("delete_object", {"key": "  "})


@pytest.mark.asyncio
async def test_rename_object_moves_path(mcp_server, fake_storage) -> None:
    fake_storage.objects.add("uploads/tmp.bin")
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "rename_object",
            {"source_key": "/uploads/tmp.bin", "target_key": "reports/final.bin"},
        )

    data = result.data
    assert data["source_key"] == "uploads/tmp.bin"
    assert data["target_key"] == "reports/final.bin"
    assert data["url"] == "https://fake-s3.test/reports/final.bin"
    assert fake_storage.renames == [("uploads/tmp.bin", "reports/final.bin")]


@pytest.mark.asyncio
async def test_rename_object_rejects_same_paths(mcp_server, fake_storage) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "rename_object",
                {"source_key": "a.bin", "target_key": "  "},
            )
    assert fake_storage.renames == []


@pytest.mark.asyncio
async def test_list_objects_returns_files_under_path(mcp_server, fake_storage) -> None:
    fake_storage.objects.update(
        {
            "images/animals/a.png",
            "images/animals/b.png",
            "images/logo/c.png",
            "uploads/x.bin",
        }
    )

    async with Client(mcp_server) as client:
        result = await client.call_tool("list_objects", {"path": "images/animals"})

    data = result.data
    # 目录语义自动补尾部斜杠，只命中该目录下对象。
    assert data["path"] == "images/animals/"
    assert data["keys"] == ["images/animals/a.png", "images/animals/b.png"]
    assert data["files"] == [
        {"key": "images/animals/a.png", "filename": "a.png", "size": 0, "last_modified": None},
        {"key": "images/animals/b.png", "filename": "b.png", "size": 0, "last_modified": None},
    ]
    assert data["count"] == 2


@pytest.mark.asyncio
async def test_list_objects_root_lists_all(mcp_server, fake_storage) -> None:
    fake_storage.objects.update({"a/1.bin", "b/2.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool("list_objects", {})

    data = result.data
    assert data["path"] == ""
    assert data["keys"] == ["a/1.bin", "b/2.bin"]
    assert data["count"] == 2


@pytest.mark.asyncio
async def test_list_objects_empty_directory(mcp_server, fake_storage) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("list_objects", {"path": "/nothing/here/"})

    data = result.data
    assert data["path"] == "nothing/here/"
    assert data["keys"] == []
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_list_objects_includes_size_metadata(mcp_server, fake_storage) -> None:
    """put 上传过的对象在列举结果中带字节数元数据。"""
    fake_storage.put(b"12345", key="uploads/sized.bin", content_type="application/octet-stream")

    async with Client(mcp_server) as client:
        result = await client.call_tool("list_objects", {"path": "uploads"})

    data = result.data
    assert data["files"][0]["size"] == 5


@pytest.mark.asyncio
async def test_move_objects_by_prefix_preserves_filenames(mcp_server, fake_storage) -> None:
    """前缀模式移动整个目录并保留相对文件名。"""
    fake_storage.objects.update({"reports/2025/a.pdf", "reports/2025/b.pdf", "reports/x.pdf"})

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "move_objects",
            {"source_path": "reports/2025", "target_path": "archive/2025"},
        )

    data = result.data
    assert data["moved_count"] == 2
    assert data["failed_count"] == 0
    assert {m["target_key"] for m in data["moved"]} == {"archive/2025/a.pdf", "archive/2025/b.pdf"}
    assert fake_storage.objects == {"reports/x.pdf", "archive/2025/a.pdf", "archive/2025/b.pdf"}
    assert data["moved"][0]["url"].startswith("https://fake-s3.test/archive/2025/")


@pytest.mark.asyncio
async def test_move_objects_with_explicit_mapping(mcp_server, fake_storage) -> None:
    """映射模式按给定 source/target 精确移动。"""
    fake_storage.objects.update({"uploads/a.bin", "uploads/b.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "move_objects",
            {
                "moves": [
                    {"source_key": "uploads/a.bin", "target_key": "keep/a.bin"},
                    {"source_key": "uploads/b.bin", "target_key": "keep/b.bin"},
                ]
            },
        )

    data = result.data
    assert data["moved_count"] == 2
    assert fake_storage.objects == {"keep/a.bin", "keep/b.bin"}


@pytest.mark.asyncio
async def test_move_objects_best_effort_reports_failures(mcp_server, fake_storage) -> None:
    """单项失败不中断，failed 汇总明细且其余对象照常移动。"""
    # 只放入 ok.bin；missing.bin 不存在，rename 会失败。
    fake_storage.objects.add("uploads/ok.bin")

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "move_objects",
            {
                "moves": [
                    {"source_key": "uploads/ok.bin", "target_key": "keep/ok.bin"},
                    {"source_key": "uploads/missing.bin", "target_key": "keep/missing.bin"},
                ]
            },
        )

    data = result.data
    assert data["moved_count"] == 1
    assert data["failed_count"] == 1
    assert data["failed"][0]["source_key"] == "uploads/missing.bin"
    assert fake_storage.objects == {"keep/ok.bin"}


@pytest.mark.asyncio
async def test_move_objects_skips_noop_pairs(mcp_server, fake_storage) -> None:
    """source 与 target 相同的项跳过而非报错。"""
    fake_storage.objects.update({"uploads/a.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "move_objects",
            {"moves": [{"source_key": "uploads/a.bin", "target_key": "uploads/a.bin"}]},
        )

    data = result.data
    assert data["moved_count"] == 0
    assert data["skipped"] == ["uploads/a.bin"]
    assert fake_storage.renames == []


@pytest.mark.asyncio
async def test_move_objects_rejects_both_modes(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "move_objects",
                {
                    "source_path": "reports/",
                    "target_path": "archive/",
                    "moves": [{"source_key": "a", "target_key": "b"}],
                },
            )


@pytest.mark.asyncio
async def test_delete_objects_by_keys(mcp_server, fake_storage) -> None:
    fake_storage.objects.update({"uploads/a.bin", "uploads/b.bin", "uploads/c.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "delete_objects",
            {"keys": ["/uploads/a.bin", "uploads/b.bin", "uploads/a.bin"]},
        )

    data = result.data
    # key 规范化 + 去重后只删两个。
    assert data["deleted"] == ["uploads/a.bin", "uploads/b.bin"]
    assert fake_storage.objects == {"uploads/c.bin"}


@pytest.mark.asyncio
async def test_delete_objects_by_path(mcp_server, fake_storage) -> None:
    fake_storage.objects.update({"tmp/a.bin", "tmp/sub/b.bin", "keep/c.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool("delete_objects", {"path": "tmp"})

    data = result.data
    assert data["deleted_count"] == 2
    assert fake_storage.objects == {"keep/c.bin"}


@pytest.mark.asyncio
async def test_delete_objects_enforces_limit_without_force(mcp_server, fake_storage) -> None:
    """超过 100 个对象时默认拒绝，force=True 才执行。"""
    fake_storage.objects.update(f"tmp/{i}.bin" for i in range(101))

    async with Client(mcp_server) as client:
        with pytest.raises(ToolError) as exc_info:
            await client.call_tool("delete_objects", {"path": "tmp"})
        assert "force" in str(exc_info.value)

        result = await client.call_tool("delete_objects", {"path": "tmp", "force": True})

    assert result.data["deleted_count"] == 101
    assert fake_storage.objects == set()


@pytest.mark.asyncio
async def test_delete_objects_best_effort_reports_failures(
    mcp_server, fake_storage, monkeypatch
) -> None:
    """单项删除失败不中断，failed 汇总明细。"""

    def failing_delete(key: str) -> None:
        if key == "uploads/bad.bin":
            raise AppError(
                ErrorCode.UPSTREAM_UNAVAILABLE, "Object storage is temporarily unavailable"
            )
        FakeObjectStorage.delete(fake_storage, key)

    monkeypatch.setattr(fake_storage, "delete", failing_delete)
    fake_storage.objects.update({"uploads/a.bin", "uploads/bad.bin"})

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "delete_objects",
            {"keys": ["uploads/a.bin", "uploads/bad.bin"]},
        )

    data = result.data
    assert data["deleted"] == ["uploads/a.bin"]
    assert data["failed"] == [
        {"key": "uploads/bad.bin", "error": "Object storage is temporarily unavailable"}
    ]


@pytest.mark.asyncio
async def test_delete_objects_rejects_both_modes(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "delete_objects",
                {"path": "tmp/", "keys": ["a"]},
            )


@pytest.mark.asyncio
async def test_download_url_preserves_signature_in_mcp_content_and_data(test_settings) -> None:
    """MCP 文本及结构化响应均保留签名 URL，不脱敏、不解码也不重编码。"""
    import json

    from agent_app.infrastructure.storage import Boto3Storage

    expected = "https://public.test/b/a%20b.jpg?Signature=fake%2B%2F%3D&Expires=123"

    class SigningClient:
        def generate_presigned_url(self, *args, **kwargs):
            return expected

    storage = Boto3Storage(client=SigningClient(), bucket="b", expires_in=300)
    server = build_mcp_server(settings=test_settings, storage=storage)
    async with Client(server) as client:
        result = await client.call_tool("get_download_url", {"key": "a b.jpg"})
    assert result.data["url"] == expected
    assert result.structured_content["url"] == expected
    assert json.loads(result.content[0].text)["url"] == expected
