"""Web 工具的进程内测试（FastMCP Client + fake 网关客户端，不联网）。"""

from typing import Any

import pytest
from fakes import FakeObjectStorage
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.errors import AppError, ErrorCode
from agent_app.tools import build_mcp_server
from agent_app.tools.web_tools import create_web_agent_tools


class FakeWebGatewayClient:
    """记录调用并可控失败的 Web 网关客户端替身。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self._error = error

    def search(self, *, query: str, limit: int = 5) -> dict[str, Any]:
        self.search_calls.append({"query": query, "limit": limit})
        if self._error is not None:
            raise self._error
        return {"results": [{"title": query}]}

    def fetch(self, *, url: str, max_chars: int = 20000) -> dict[str, Any]:
        self.fetch_calls.append({"url": url, "max_chars": max_chars})
        if self._error is not None:
            raise self._error
        return {"content": f"body of {url}"}


def _settings_with_gateway() -> Settings:
    """构造带完整 web_gateway 节的最小应用配置。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "web_gateway": {
                "base_url": "https://surf.leegoo.ltd",
                "api_token": "gateway-token",
            },
        }
    )


@pytest.fixture
def fake_client() -> FakeWebGatewayClient:
    """记录调用的默认成功网关替身。"""
    return FakeWebGatewayClient()


@pytest.fixture
def mcp_server(test_settings, fake_client):
    """注入 fake 存储与 fake 网关的聚合 MCP 服务实例。"""
    return build_mcp_server(
        settings=test_settings,
        storage=FakeObjectStorage(),
        web_gateway=fake_client,
    )


@pytest.mark.asyncio
async def test_mcp_lists_web_tools_when_gateway_configured() -> None:
    server = build_mcp_server(
        settings=_settings_with_gateway(),
        storage=FakeObjectStorage(),
        web_gateway=FakeWebGatewayClient(),
    )

    async with Client(server) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert {"web_search", "web_fetch"} <= names


@pytest.mark.asyncio
async def test_mcp_omits_web_tools_when_gateway_missing(test_settings) -> None:
    server = build_mcp_server(settings=test_settings, storage=FakeObjectStorage())

    async with Client(server) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert "web_search" not in names
    assert "web_fetch" not in names


@pytest.mark.asyncio
async def test_mcp_search_passes_normalized_query(mcp_server, fake_client) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool("web_search", {"query": "  langgraph  "})

    assert result.data == {"results": [{"title": "langgraph"}]}
    assert fake_client.search_calls == [{"query": "langgraph", "limit": 5}]


@pytest.mark.asyncio
async def test_mcp_fetch_passes_url_and_max_chars(mcp_server, fake_client) -> None:
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "web_fetch",
            {"url": "https://example.com/page", "max_chars": 3000},
        )

    assert result.data == {"content": "body of https://example.com/page"}
    assert fake_client.fetch_calls == [{"url": "https://example.com/page", "max_chars": 3000}]


@pytest.mark.asyncio
async def test_mcp_search_requires_non_blank_query(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_search", {"query": "   "})


@pytest.mark.asyncio
async def test_mcp_search_rejects_out_of_range_limit(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_search", {"query": "x", "limit": 0})
        with pytest.raises(ToolError):
            await client.call_tool("web_search", {"query": "x", "limit": 11})


@pytest.mark.asyncio
async def test_mcp_fetch_requires_non_blank_url(mcp_server) -> None:
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_fetch", {"url": ""})


@pytest.mark.asyncio
async def test_mcp_wraps_gateway_error_as_tool_error() -> None:
    failing = FakeWebGatewayClient(
        error=AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "Web gateway is temporarily unavailable")
    )
    server = build_mcp_server(
        settings=_settings_with_gateway(),
        storage=FakeObjectStorage(),
        web_gateway=failing,
    )

    async with Client(server) as client:
        with pytest.raises(ToolError, match="temporarily unavailable"):
            await client.call_tool("web_search", {"query": "x"})


def test_agent_tools_names_match_contract(fake_client) -> None:
    tools = create_web_agent_tools(fake_client)

    assert {tool.name for tool in tools} == {"web_search", "web_fetch"}


def test_agent_search_returns_gateway_payload(fake_client) -> None:
    web_search = create_web_agent_tools(fake_client)[0]

    result = web_search.invoke({"query": "  langgraph  ", "limit": 7})

    assert result == {"results": [{"title": "langgraph"}]}
    assert fake_client.search_calls == [{"query": "langgraph", "limit": 7}]


def test_agent_fetch_returns_gateway_payload(fake_client) -> None:
    web_fetch = create_web_agent_tools(fake_client)[1]

    result = web_fetch.invoke({"url": "https://example.com/page", "max_chars": 3000})

    assert result == {"content": "body of https://example.com/page"}
    assert fake_client.fetch_calls == [{"url": "https://example.com/page", "max_chars": 3000}]


def test_agent_tools_return_error_dict_instead_of_raising() -> None:
    """网关故障时 Agent 工具必须返回错误字典而非抛异常（抛异常会中断整个任务）。"""
    failing = FakeWebGatewayClient(
        error=AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "Web gateway is temporarily unavailable")
    )
    tools = create_web_agent_tools(failing)

    for tool_instance, args in (
        (tools[0], {"query": "x"}),
        (tools[1], {"url": "https://example.com/page"}),
    ):
        result = tool_instance.invoke(args)
        assert result == {
            "error": {
                "code": "UPSTREAM_UNAVAILABLE",
                "message": "Web gateway is temporarily unavailable",
            }
        }


def test_agent_tools_validate_args(fake_client) -> None:
    """args_schema 校验失败在工具边界抛出，不会静默放行到网关。"""
    from pydantic import ValidationError

    web_search = create_web_agent_tools(fake_client)[0]

    with pytest.raises(ValidationError, match="query is required"):
        web_search.invoke({"query": "   "})

    assert fake_client.search_calls == []
