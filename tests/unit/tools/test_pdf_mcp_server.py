"""PDF MCP 服务的进程内测试（FastMCP Client，不起 HTTP 端口、不联网）。"""

import base64
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from agent_app.tools import build_mcp_server
from agent_app.tools.pdf.persist import STATIC_URL_PREFIX


@pytest.fixture
def mcp_server(tmp_path: Path):
    """挂载到临时目录的聚合 MCP 服务实例。"""
    return build_mcp_server(output_dir=tmp_path, url_prefix=STATIC_URL_PREFIX)


@pytest.mark.asyncio
async def test_tool_lists_pdf_to_image(mcp_server) -> None:
    async with Client(mcp_server) as client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["pdf_to_image"]


@pytest.mark.asyncio
async def test_tool_base64_mode_returns_inline_data(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("pdf_to_image", {"source": pdf_source_url})

    data = result.data
    assert data["page_count"] == 3
    assert [img["page"] for img in data["images"]] == [1]
    img = data["images"][0]
    assert set(img.keys()) == {"page", "data", "width", "height"}
    assert base64.b64decode(img["data"])[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_tool_url_mode_persists_files(mcp_server, pdf_source_url, tmp_path) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "pdf_to_image", {"source": pdf_source_url, "return_mode": "url"}
        )

    url = result.data["images"][0]["url"]
    assert url.startswith(f"{STATIC_URL_PREFIX}/")
    relative = url.split(f"{STATIC_URL_PREFIX}/", 1)[1]
    assert (tmp_path / relative).exists()


@pytest.mark.asyncio
async def test_tool_accepts_pages_and_ranges(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "pdf_to_image",
            {"source": pdf_source_url, "pages": [1], "page_ranges": [[2, 3]]},
        )

    assert [img["page"] for img in result.data["images"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_tool_validates_dpi_bounds(mcp_server, pdf_source_url) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("pdf_to_image", {"source": pdf_source_url, "dpi": 700})


@pytest.mark.asyncio
async def test_tool_requires_source(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("pdf_to_image", {})
