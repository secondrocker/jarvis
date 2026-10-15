"""网络搜索 / 网页阅读工具的进程内测试（FastMCP Client，不联网）。

上游是智谱 Coding Plan 的 Remote MCP Server，测试用 ``Client`` / ``StreamableHttpTransport``
替身拦截，断言请求参数与响应整形，不产生真实网络调用。
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools import build_mcp_server, web_search_tools
from agent_app.tools.web_search_tools import _decode_payload, create_web_agent_tools

_SEARCH_PAYLOAD = [
    {
        "title": "结果一",
        "link": "https://a.test/1",
        "content": "摘要一",
        "refer": "ref_1",
    },
    {
        "title": "结果二",
        "link": "https://a.test/2",
        "content": "摘要二",
        "refer": "ref_2",
    },
]

_READER_PAYLOAD = {
    "title": "页面标题",
    "url": "https://example.com/page",
    "content": "# 正文",
    "metadata": {"description": "页面描述", "lang": "zh"},
    "external": {},
}


class _FakeTransport:
    """StreamableHttpTransport 替身：记录 url / headers / 超时工厂。"""

    instances: list["_FakeTransport"] = []

    def __init__(self, url, *, headers=None, httpx_client_factory=None, **_kwargs) -> None:  # noqa: ANN001
        self.url = url
        self.headers = headers
        self.httpx_client_factory = httpx_client_factory
        _FakeTransport.instances.append(self)


class _FakeClient:
    """fastmcp Client 替身：按工具名返回预置载荷（与上游一致地双层序列化）。"""

    responses: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    error: Exception | None = None
    delay: float = 0.0

    def __init__(self, transport: _FakeTransport) -> None:
        self.transport = transport

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if _FakeClient.error is not None:
            raise _FakeClient.error
        if _FakeClient.delay:
            await asyncio.sleep(_FakeClient.delay)
        _FakeClient.calls.append({"tool": name, "arguments": arguments})
        # 上游把结果再序列化一层：content 文本形如 "[{\"title\": ...}]"。
        text = json.dumps(json.dumps(_FakeClient.responses[name], ensure_ascii=False))
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


@pytest.fixture
def patch_zhipu(monkeypatch):
    """patch 上游 MCP 客户端；返回 (捕获的请求列表, transport 列表)。"""
    _FakeClient.calls = []
    _FakeClient.responses = {
        web_search_tools.SEARCH_TOOL_NAME: _SEARCH_PAYLOAD,
        web_search_tools.READER_TOOL_NAME: _READER_PAYLOAD,
    }
    _FakeClient.error = None
    _FakeClient.delay = 0.0
    _FakeTransport.instances = []
    monkeypatch.setattr(web_search_tools, "Client", _FakeClient)
    monkeypatch.setattr(web_search_tools, "StreamableHttpTransport", _FakeTransport)
    return _FakeClient.calls


@pytest.fixture
def zhipu_settings() -> Settings:
    """带智谱 Coding Plan MCP 配置的应用配置。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "zhipu_web": {"api_key": SecretStr("sk-zhipu")},
        }
    )


@pytest.mark.asyncio
async def test_mcp_web_search_roundtrip(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        result = await client.call_tool(
            "web_search",
            {"query": "  langgraph 教程  ", "count": 1, "recency": "oneWeek"},
        )

    data = result.data
    assert data["query"] == "langgraph 教程"
    # 上游单次约 10 条，count 只在本地截断。
    assert data["count"] == 1
    assert [item["link"] for item in data["results"]] == ["https://a.test/1"]
    assert data["results"][0]["refer"] == "ref_1"

    call = patch_zhipu[0]
    assert call["tool"] == "web_search_prime"
    assert call["arguments"] == {
        "search_query": "langgraph 教程",
        "search_recency_filter": "oneWeek",
        "content_size": "medium",
        "location": "cn",
    }
    transport = _FakeTransport.instances[0]
    assert transport.url == "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp"
    assert transport.headers == {"Authorization": "Bearer sk-zhipu"}
    assert transport.httpx_client_factory is not None


@pytest.mark.asyncio
async def test_mcp_web_search_domain_and_overrides(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        await client.call_tool(
            "web_search",
            {
                "query": "x",
                "domain": " www.example.com ",
                "content_size": "high",
                "location": "us",
            },
        )
    assert patch_zhipu[0]["arguments"]["search_domain_filter"] == "www.example.com"
    assert patch_zhipu[0]["arguments"]["content_size"] == "high"
    assert patch_zhipu[0]["arguments"]["location"] == "us"


@pytest.mark.asyncio
async def test_mcp_web_fetch_roundtrip(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        result = await client.call_tool(
            "web_fetch", {"url": "https://example.com/page", "return_format": "text"}
        )

    data = result.data
    assert data["title"] == "页面标题"
    assert data["description"] == "页面描述"
    assert data["content"] == "# 正文"
    call = patch_zhipu[0]
    assert call["tool"] == "webReader"
    assert call["arguments"]["url"] == "https://example.com/page"
    assert call["arguments"]["return_format"] == "text"
    # 上游自身超时须先于本地硬超时触发（fetch_timeout_seconds 默认 40）。
    assert call["arguments"]["timeout"] == 35
    transport = _FakeTransport.instances[0]
    assert transport.url == "https://open.bigmodel.cn/api/mcp/web_reader/mcp"
    assert transport.headers == {"Authorization": "Bearer sk-zhipu"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"query": "   "},
        {"query": "x" * 71},
        {"query": "x", "count": 0},
        {"query": "x", "count": 51},
        {"query": "x", "recency": "yesterday"},
        {"query": "x", "content_size": "huge"},
        {"query": "x", "location": "mars"},
    ],
)
async def test_mcp_web_search_rejects_invalid_args(zhipu_settings, args) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_search", args)


@pytest.mark.asyncio
async def test_schema_documents_query_limit(zhipu_settings) -> None:
    """参数说明必须写明 70 字符上限，调用方无需撞报错才知道。"""
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    search_props = tools["web_search"].inputSchema["properties"]
    assert "70" in search_props["query"]["description"]
    assert "拆成多次" in search_props["query"]["description"]
    # 其余参数也有说明，避免调用方靠猜。
    for name in ("count", "recency", "domain", "content_size", "location"):
        assert search_props[name].get("description")
    fetch_props = tools["web_fetch"].inputSchema["properties"]
    assert fetch_props["url"].get("description")


@pytest.mark.asyncio
async def test_over_long_query_error_is_actionable(zhipu_settings) -> None:
    """超长 query 报错需给出上限、当前长度与改写建议（而非 pydantic 英文原文）。"""
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("web_search", {"query": "长" * 71})
    message = str(excinfo.value)
    assert "最多 70 字符" in message
    assert "当前 71 字符" in message
    assert "拆成多次" in message
    assert "should have at most" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,expected",
    [
        ({"query": "x", "count": 51}, "超出范围"),
        ({"query": "   "}, "不能为空"),
    ],
)
async def test_invalid_args_error_is_actionable(zhipu_settings, args, expected) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError, match=expected):
            await client.call_tool("web_search", args)


@pytest.mark.asyncio
async def test_mcp_web_fetch_rejects_non_http_url(zhipu_settings) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_fetch", {"url": "ftp://example.com"})


@pytest.mark.asyncio
async def test_mcp_upstream_error_surfaces_message(zhipu_settings, patch_zhipu) -> None:
    _FakeClient.error = ToolError("MCP error -400: search_query cannot be empty")
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError, match="search_query cannot be empty"):
            await client.call_tool("web_search", {"query": "x"})


@pytest.mark.asyncio
async def test_mcp_upstream_timeout_surfaces_hint(patch_zhipu) -> None:
    """上游无响应（多为 Key 无效）时必须被本地超时兜住，而不是一直挂住。"""
    settings = Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "zhipu_web": {"api_key": SecretStr("sk-zhipu"), "search_timeout_seconds": 0.05},
        }
    )
    _FakeClient.delay = 5.0
    async with Client(build_mcp_server(settings=settings, storage=object())) as client:
        with pytest.raises(ToolError, match="timed out"):
            await client.call_tool("web_search", {"query": "x"})


@pytest.mark.asyncio
async def test_tools_omitted_without_api_key() -> None:
    settings = Settings.model_validate(
        {"openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"}}
    )
    async with Client(build_mcp_server(settings=settings, storage=object())) as client:
        tools = await client.list_tools()
    names = {tool.name for tool in tools}
    assert "web_search" not in names
    assert "web_fetch" not in names


@pytest.mark.asyncio
async def test_tools_omitted_with_blank_api_key() -> None:
    """占位空串 api_key 视为未配置。"""
    settings = Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "zhipu_web": {"api_key": ""},
        }
    )
    assert settings.zhipu_web.enabled is False


@pytest.mark.asyncio
async def test_agent_face_returns_error_dict(monkeypatch, patch_zhipu) -> None:
    """Agent 面：上游失败返回 {"error": ...} 字典而非抛异常。"""
    _FakeClient.error = ToolError("MCP error -500: 500 Internal Server Error")
    from agent_app.config import ZhipuWebConfig

    tools = create_web_agent_tools(ZhipuWebConfig(api_key=SecretStr("sk-zhipu")))
    by_name = {tool.name: tool for tool in tools}
    result = await by_name["web_search"].ainvoke({"query": "x"})
    assert result["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    result = await by_name["web_fetch"].ainvoke({"url": "not-a-url"})
    assert result["error"]["code"] == "validation"


def test_decode_payload_handles_double_encoded_json() -> None:
    """上游把结果再序列化一层，需连解两次。"""
    assert _decode_payload(json.dumps(json.dumps(_SEARCH_PAYLOAD))) == _SEARCH_PAYLOAD
    # 单层编码同样兼容。
    assert _decode_payload(json.dumps(_SEARCH_PAYLOAD)) == _SEARCH_PAYLOAD
    # 非 JSON 文本原样返回，不抛异常。
    assert _decode_payload("plain text") == "plain text"


@pytest.mark.asyncio
async def test_mcp_urls_validated() -> None:
    """MCP 端点必须是 http(s)，便于配置写错时尽早失败。"""
    with pytest.raises(ValueError):
        Settings.model_validate(
            {
                "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
                "zhipu_web": {"api_key": SecretStr("sk-zhipu"), "search_mcp_url": "ftp://x"},
            }
        )
