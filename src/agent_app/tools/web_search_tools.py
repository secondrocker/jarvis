"""基于智谱 BigModel 工具 API 的网络搜索 / 网页阅读工具。

同一份实现提供两面暴露（与既有工具装配哲学一致）：
- MCP 面（register_web_search_tools）：AppError 转为 ToolError 抛出；
- Agent 面（create_web_agent_tools）：AppError 转为 {"error": {...}} 字典返回，
  供 Deep Agent 作为普通 LangChain 工具装配。

不做任何用量限制（按需求明确要求）。
"""

from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from pydantic import Field as PDField

from agent_app.config import ZhipuWebConfig
from agent_app.errors import AppError, ErrorCode
from agent_app.schemas.web_search_tools import WebFetchInput, WebSearchInput


def _post_zhipu(config: ZhipuWebConfig, path: str, payload: dict[str, Any], timeout: float) -> dict:
    """调用 BigModel 工具 API 并返回 JSON 体。

    异常:
        AppError: 网络层失败、非 2xx 或响应非 JSON 时抛出 UPSTREAM_UNAVAILABLE。
    """
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",  # type: ignore[union-attr]
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            f"{config.base_url}{path}",  # type: ignore[operator]
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Zhipu web tool service is temporarily unavailable",
        ) from error
    if response.status_code >= 400:
        # 错误体为 {error: {code, message}}；附上 message 便于调用方修正
        # query/url，仍不泄露密钥等内部信息。
        detail = "upstream error"
        try:
            detail = str(response.json().get("error", {}).get("message") or detail)
        except ValueError:
            pass
        raise AppError(ErrorCode.UPSTREAM_UNAVAILABLE, f"Zhipu web tool failed: {detail}")
    try:
        return response.json()
    except ValueError as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Zhipu web tool returned an unexpected response",
        ) from error


def _search(
    config: ZhipuWebConfig,
    query: str,
    count: int,
    search_engine: str | None,
    recency: str,
    domain: str | None,
) -> dict[str, Any]:
    """执行网络搜索并整形为稳定结构。"""
    payload: dict[str, Any] = {
        "search_query": query,
        "search_engine": search_engine or config.search_engine,
        "search_intent": False,
        "count": count,
        "search_recency_filter": recency,
        "content_size": config.content_size,
    }
    if domain:
        payload["search_domain_filter"] = domain
    body = _post_zhipu(config, "/paas/v4/web_search", payload, config.search_timeout_seconds)
    results = [
        {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "content": item.get("content", ""),
            "media": item.get("media", ""),
            "publish_date": item.get("publish_date", ""),
        }
        for item in body.get("search_result") or []
    ]
    return {"query": query, "count": len(results), "results": results}


def _fetch(config: ZhipuWebConfig, url: str, return_format: str | None) -> dict[str, Any]:
    """执行网页阅读并整形为稳定结构。"""
    payload: dict[str, Any] = {
        "url": url,
        "return_format": return_format or config.return_format,
        "timeout": int(config.fetch_timeout_seconds),
    }
    body = _post_zhipu(config, "/paas/v4/reader", payload, config.fetch_timeout_seconds + 10.0)
    result = body.get("reader_result") or {}
    return {
        "url": url,
        "title": result.get("title", ""),
        "description": result.get("description", ""),
        "content": result.get("content", ""),
    }


def register_web_search_tools(mcp: FastMCP, *, config: ZhipuWebConfig) -> None:
    """把网络搜索 / 网页阅读 tool 注册到给定 MCP 服务（MCP 面）。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        config: 智谱工具 API 配置（api_key 必须已配置）。
    """

    @mcp.tool
    def web_search(
        query: str,
        count: int = 10,
        search_engine: str | None = None,
        recency: str = "noLimit",
        domain: str | None = None,
    ) -> dict[str, Any]:
        """网络搜索：返回与 query 相关的网页列表（标题、链接、摘要、来源、发布时间）。

        query 为搜索关键词（不超过 70 字符）；count 为结果条数（1-50，默认 10）；
        search_engine 可选（search_std 默认 / search_pro / search_pro_sogou /
        search_pro_quark）；recency 可选（oneDay / oneWeek / oneMonth / oneYear /
        noLimit 默认）；domain 可选，限定白名单域名（如 www.example.com）。

        返回值:
            含 query、count、results 的字典；results 每项含 title、link、
            content、media、publish_date。
        """
        try:
            params = WebSearchInput(
                query=query,
                count=count,
                search_engine=search_engine,  # type: ignore[arg-type]
                recency=recency,  # type: ignore[arg-type]
                domain=domain,
            )
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid search parameters"
            raise ToolError(str(detail)) from error
        try:
            return _search(
                config,
                params.query,
                params.count,
                params.search_engine,
                params.recency,
                params.domain,
            )
        except AppError as error:
            raise ToolError(error.public_message) from error

    @mcp.tool
    def web_fetch(url: str, return_format: str | None = None) -> dict[str, Any]:
        """网页阅读：抓取并解析指定 URL 的正文，返回 markdown（默认）或纯文本。

        url 必须是 http(s) 链接；return_format 可选（markdown 默认 / text）。

        返回值:
            含 url、title、description、content 的字典；content 为解析后的正文。
        """
        try:
            params = WebFetchInput(url=url, return_format=return_format)  # type: ignore[arg-type]
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid fetch parameters"
            raise ToolError(str(detail)) from error
        try:
            return _fetch(config, params.url, params.return_format)
        except AppError as error:
            raise ToolError(error.public_message) from error


class _WebSearchArgs(BaseModel):
    """Agent 面 web_search 的参数模型。"""

    query: str = PDField(min_length=1)
    count: int = PDField(default=10, ge=1, le=50)
    search_engine: str | None = None
    recency: str = "noLimit"
    domain: str | None = None


class _WebFetchArgs(BaseModel):
    """Agent 面 web_fetch 的参数模型。"""

    url: str = PDField(min_length=1)
    return_format: str | None = None


def create_web_agent_tools(config: ZhipuWebConfig) -> list[BaseTool]:
    """构造 Deep Agent 自用的搜索/阅读工具（Agent 面，错误以字典返回）。

    参数:
        config: 智谱工具 API 配置（api_key 必须已配置）。

    返回值:
        含 web_search 与 web_fetch 两个工具的列表。
    """

    async def web_search(query: str, count: int = 10, **kwargs: Any) -> dict[str, Any]:
        """网络搜索：给定关键词返回网页列表；已知具体 URL 时改用 web_fetch。"""
        try:
            params = WebSearchInput(query=query, count=count, **kwargs)
        except ValidationError as error:
            return {"error": {"code": "validation", "message": error.errors()[0]["msg"]}}
        try:
            return _search(
                config,
                params.query,
                params.count,
                params.search_engine,
                params.recency,
                params.domain,
            )
        except AppError as error:
            return {"error": {"code": error.code.value, "message": error.public_message}}

    async def web_fetch(url: str, return_format: str | None = None) -> dict[str, Any]:
        """网页阅读：抓取并解析指定 URL 的正文（markdown）。"""
        try:
            params = WebFetchInput(url=url, return_format=return_format)
        except ValidationError as error:
            return {"error": {"code": "validation", "message": error.errors()[0]["msg"]}}
        try:
            return _fetch(config, params.url, params.return_format)
        except AppError as error:
            return {"error": {"code": error.code.value, "message": error.public_message}}

    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description=(
                "网络搜索：给定关键词返回网页列表（标题/链接/摘要/来源/发布时间）。"
                "已知具体 URL 时改用 web_fetch。query ≤70 字符，count 1-50。"
            ),
            args_schema=_WebSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=web_fetch,
            name="web_fetch",
            description="网页阅读：抓取并解析指定 URL 的正文，返回 title/description/content。",
            args_schema=_WebFetchArgs,
        ),
    ]
