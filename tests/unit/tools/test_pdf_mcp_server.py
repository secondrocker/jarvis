"""PDF MCP 服务的进程内测试（FastMCP Client，不起 HTTP 端口、不联网）。"""

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError

from agent_app.tools import build_mcp_server


@pytest.fixture
def mcp_server():
    """注入内存 fake 存储的聚合 MCP 服务实例。"""
    return build_mcp_server(storage=FakeObjectStorage())


@pytest.mark.asyncio
async def test_tool_lists_pdf_to_image(mcp_server) -> None:
    async with Client(mcp_server) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert "pdf_to_image" in names
    assert {"upload_from_url", "get_download_url", "get_upload_url"} <= names


@pytest.mark.asyncio
async def test_tool_returns_download_urls(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("pdf_to_image", {"url": pdf_source_url})

    data = result.data
    assert data["page_count"] == 3
    assert [img["page"] for img in data["images"]] == [1]
    img = data["images"][0]
    assert set(img.keys()) == {"page", "url", "width", "height"}
    assert img["url"].startswith("https://fake-s3.test/pdf_images/")
    assert img["url"].endswith("/page_1.png")


@pytest.mark.asyncio
async def test_tool_accepts_pages_and_ranges(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "pdf_to_image",
            {"url": pdf_source_url, "pages": [1], "page_ranges": [[2, 3]]},
        )

    assert [img["page"] for img in result.data["images"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_tool_validates_dpi_bounds(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("pdf_to_image", {"url": pdf_source_url, "dpi": 700})


@pytest.mark.asyncio
async def test_tool_requires_url(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("pdf_to_image", {})
