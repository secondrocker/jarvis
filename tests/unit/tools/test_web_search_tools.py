"""网络搜索 / 网页阅读工具的进程内测试（FastMCP Client，不联网）。"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import SecretStr

from agent_app.config import Settings
from agent_app.tools import build_mcp_server, web_search_tools
from agent_app.tools.web_search_tools import create_web_agent_tools


class _FakeResponse:
    """httpx 响应替身：固定状态码与 JSON 体。"""

    def __init__(self, *, json_body: dict | None = None, status_code: int = 200) -> None:
        self._json = json_body
        self.status_code = status_code

    def json(self) -> dict:
        return self._json  # type: ignore[return-value]


@pytest.fixture
def zhipu_settings() -> Settings:
    """带智谱工具 API 配置的应用配置。"""
    return Settings.model_validate(
        {
            "openai": {"api_key": SecretStr("test-key"), "model": "gpt-4o-mini"},
            "zhipu_web": {
                "base_url": "https://open.bigmodel.cn/api/",
                "api_key": SecretStr("sk-zhipu"),
            },
        }
    )


@pytest.fixture
def patch_zhipu(monkeypatch):
    """patch httpx.post 捕获请求；按端点返回搜索/阅读的典型响应。"""
    captured: list[dict] = []

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        captured.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/web_search"):
            return _FakeResponse(
                json_body={
                    "id": "1",
                    "search_result": [
                        {
                            "title": "结果一",
                            "link": "https://a.test/1",
                            "content": "摘要",
                            "media": "A站",
                            "publish_date": "2026-10-01",
                        }
                    ],
                }
            )
        return _FakeResponse(
            json_body={
                "reader_result": {
                    "title": "页面标题",
                    "description": "页面描述",
                    "content": "# 正文",
                    "url": json["url"],
                }
            }
        )

    monkeypatch.setattr(web_search_tools.httpx, "post", fake_post)
    return captured


@pytest.mark.asyncio
async def test_mcp_web_search_roundtrip(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        result = await client.call_tool(
            "web_search",
            {"query": "  langgraph 教程  ", "count": 5, "recency": "oneWeek"},
        )

    data = result.data
    assert data["query"] == "langgraph 教程"
    assert data["results"][0]["link"] == "https://a.test/1"
    body = patch_zhipu[0]
    assert body["url"] == "https://open.bigmodel.cn/api/paas/v4/web_search"
    assert body["json"]["search_query"] == "langgraph 教程"
    assert body["json"]["search_engine"] == "search_std"
    assert body["json"]["count"] == 5
    assert body["json"]["search_recency_filter"] == "oneWeek"
    assert body["json"]["search_intent"] is False
    assert body["headers"]["Authorization"] == "Bearer sk-zhipu"


@pytest.mark.asyncio
async def test_mcp_web_search_domain_and_engine_override(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        await client.call_tool(
            "web_search",
            {"query": "x", "search_engine": "search_pro_sogou", "domain": " www.example.com "},
        )
    body = patch_zhipu[0]
    assert body["json"]["search_engine"] == "search_pro_sogou"
    assert body["json"]["search_domain_filter"] == "www.example.com"


@pytest.mark.asyncio
async def test_mcp_web_fetch_roundtrip(zhipu_settings, patch_zhipu) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        result = await client.call_tool(
            "web_fetch", {"url": "https://example.com/page", "return_format": "text"}
        )

    data = result.data
    assert data["title"] == "页面标题"
    assert data["content"] == "# 正文"
    body = patch_zhipu[0]
    assert body["url"] == "https://open.bigmodel.cn/api/paas/v4/reader"
    assert body["json"]["url"] == "https://example.com/page"
    assert body["json"]["return_format"] == "text"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"query": "   "},
        {"query": "x" * 71},
        {"query": "x", "count": 0},
        {"query": "x", "count": 51},
        {"query": "x", "recency": "yesterday"},
    ],
)
async def test_mcp_web_search_rejects_invalid_args(zhipu_settings, args) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_search", args)


@pytest.mark.asyncio
async def test_mcp_web_fetch_rejects_non_http_url(zhipu_settings) -> None:
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_fetch", {"url": "ftp://example.com"})


@pytest.mark.asyncio
async def test_mcp_upstream_error_surfaces_message(zhipu_settings, monkeypatch) -> None:
    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        return _FakeResponse(
            status_code=429,
            json_body={"error": {"code": "1701", "message": "并发已达上限"}},
        )

    monkeypatch.setattr(web_search_tools.httpx, "post", fake_post)
    async with Client(build_mcp_server(settings=zhipu_settings, storage=object())) as client:
        with pytest.raises(ToolError, match="并发已达上限"):
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
async def test_agent_face_returns_error_dict(monkeypatch) -> None:
    """Agent 面：上游失败返回 {"error": ...} 字典而非抛异常。"""

    def fake_post(url, *, json, headers, timeout):  # noqa: ANN001
        return _FakeResponse(
            status_code=503, json_body={"error": {"code": "1302", "message": "bad"}}
        )

    monkeypatch.setattr(web_search_tools.httpx, "post", fake_post)
    from agent_app.config import ZhipuWebConfig

    tools = create_web_agent_tools(ZhipuWebConfig(api_key=SecretStr("sk-zhipu")))
    by_name = {tool.name: tool for tool in tools}
    result = await by_name["web_search"].ainvoke({"query": "x"})
    assert result["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    result = await by_name["web_fetch"].ainvoke({"url": "not-a-url"})
    assert result["error"]["code"] == "validation"
