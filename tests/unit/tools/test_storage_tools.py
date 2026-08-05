"""对象存储 MCP 工具的进程内测试（FastMCP Client，不起 HTTP 端口、不联网）。"""

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError

from agent_app.tools import build_mcp_server, storage_tools


class _FakeResponse:
    """httpx.get 的最小替身，返回预设字节。"""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def mcp_server():
    """注入内存 fake 存储的聚合 MCP 服务实例。"""
    return build_mcp_server(storage=FakeObjectStorage())


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
async def test_upload_from_url_uses_custom_prefix(mcp_server, patch_download) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "upload_from_url",
            {
                "source_url": "https://src.test/file",
                "content_type": "application/pdf",
                "key_prefix": "/reports/",
            },
        )

    assert result.data["key"].startswith("reports/")
    assert result.data["url"].startswith("https://fake-s3.test/reports/")


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
async def test_get_upload_url_uses_custom_prefix(mcp_server) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_upload_url",
            {"content_type": "application/pdf", "key_prefix": "reports"},
        )

    assert result.data["key"].startswith("reports/")
    assert result.data["url"].startswith("https://fake-s3.test/reports/")


@pytest.mark.asyncio
async def test_get_upload_url_requires_content_type(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_upload_url", {"content_type": "  "})
