"""基于智谱 GLM Coding Plan 专属 MCP 的网络搜索 / 网页阅读工具。

上游是编程套餐的 Remote MCP Server（`web_search_prime` / `webReader`），走 MCP
协议而非开放平台的 `/paas/v4` REST 工具 API——两者是独立的用量池，套餐额度只在
MCP 侧生效。

同一份实现提供两面暴露（与既有工具装配哲学一致）：
- MCP 面（register_web_search_tools）：AppError 转为 ToolError 抛出；
- Agent 面（create_web_agent_tools）：AppError 转为 {"error": {...}} 字典返回，
  供 Deep Agent 作为普通 LangChain 工具装配。

不做任何用量限制（按需求明确要求）。
"""

import asyncio
import json
from typing import Annotated, Any

import httpx
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from langchain_core.tools import BaseTool
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ValidationError
from pydantic import Field as PDField

from agent_app.config import ZhipuWebConfig
from agent_app.errors import AppError, ErrorCode
from agent_app.schemas.web_search_tools import WebFetchInput, WebSearchInput

# 上游 MCP Server 暴露的工具名（注意：文档写的是 webSearchPrime，实际注册为下划线形式）。
SEARCH_TOOL_NAME = "web_search_prime"
READER_TOOL_NAME = "webReader"


def _auth_headers(config: ZhipuWebConfig) -> dict[str, str]:
    """构造上游 MCP 的鉴权请求头。"""
    return {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}  # type: ignore[union-attr]


def _httpx_client_factory(timeout_seconds: float):
    """构造带硬超时的 httpx 客户端工厂。

    Key 无效 / 未开通套餐时上游会返回非 JSON-RPC 响应，MCP 客户端可能长时间
    等待，因此必须给底层 HTTP 客户端设超时（外层另有 asyncio 超时兜底）。
    """

    def factory(
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = True,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
        )

    return factory


def _decode_payload(raw: str) -> Any:
    """解析上游 tool 返回的文本。

    上游把结果对象再序列化成字符串放进 content 文本里（形如
    `"[{\\"title\\": ...}]"`），故最多连解两层；解不出 JSON 时原样返回。
    """
    value: Any = raw
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _extract_text(result: Any) -> str:
    """取出 MCP 工具结果中的文本载荷。"""
    texts = [
        block.text
        for block in getattr(result, "content", None) or []
        if getattr(block, "text", None)
    ]
    if not texts:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Zhipu web tool returned an unexpected response",
        )
    return texts[0]


async def _call_mcp_tool(
    config: ZhipuWebConfig,
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    """调用上游 MCP Server 的指定 tool 并返回解码后的结果。

    每次调用建立独立会话（无状态、无跨请求生命周期问题）。

    异常:
        AppError: 超时、协议层失败或工具报错时抛出 UPSTREAM_UNAVAILABLE。
    """

    async def invoke() -> Any:
        transport = StreamableHttpTransport(
            url,
            headers=_auth_headers(config),
            httpx_client_factory=_httpx_client_factory(timeout_seconds),
        )
        async with Client(transport) as client:
            result = await client.call_tool(tool_name, arguments)
        return _decode_payload(_extract_text(result))

    try:
        return await asyncio.wait_for(invoke(), timeout_seconds)
    except TimeoutError as error:
        # 套餐 Key 无效 / 未开通时上游会返回非 JSON-RPC 响应，表现为一直等待。
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            f"Zhipu web tool timed out after {timeout_seconds:g}s "
            "(check zhipu_web.api_key is a GLM Coding Plan key)",
        ) from error
    except (ToolError, McpError) as error:
        raise AppError(ErrorCode.UPSTREAM_UNAVAILABLE, f"Zhipu web tool failed: {error}") from error
    except httpx.HTTPError as error:
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "Zhipu web tool service is temporarily unavailable",
        ) from error
    except AppError:
        raise
    except Exception as error:  # noqa: BLE001 - MCP 客户端异常种类不稳定，统一降级
        raise AppError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            f"Zhipu web tool failed: {error or type(error).__name__}",
        ) from error


async def _search(
    config: ZhipuWebConfig,
    query: str,
    count: int,
    recency: str,
    domain: str | None,
    content_size: str | None,
    location: str | None,
) -> dict[str, Any]:
    """执行网络搜索并整形为稳定结构。"""
    payload: dict[str, Any] = {
        "search_query": query,
        "search_recency_filter": recency,
        "content_size": content_size or config.content_size,
        "location": location or config.location,
    }
    if domain:
        payload["search_domain_filter"] = domain
    items = await _call_mcp_tool(
        config,
        config.search_mcp_url,
        SEARCH_TOOL_NAME,
        payload,
        config.search_timeout_seconds,
    )
    # 上游不接受条数参数（单次约 10 条），count 仅做本地截断。
    results = [
        {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "content": item.get("content", ""),
            "refer": item.get("refer", ""),
        }
        for item in (items or [])
        if isinstance(item, dict)
    ][:count]
    return {"query": query, "count": len(results), "results": results}


async def _fetch(config: ZhipuWebConfig, url: str, return_format: str | None) -> dict[str, Any]:
    """执行网页阅读并整形为稳定结构。"""
    payload: dict[str, Any] = {
        "url": url,
        "return_format": return_format or config.return_format,
        # 上游自身超时应先于本地硬超时触发，留出余量。
        "timeout": max(1, int(config.fetch_timeout_seconds) - 5),
    }
    payload_result = await _call_mcp_tool(
        config,
        config.reader_mcp_url,
        READER_TOOL_NAME,
        payload,
        config.fetch_timeout_seconds,
    )
    result = payload_result if isinstance(payload_result, dict) else {}
    metadata = result.get("metadata") or {}
    return {
        "url": result.get("url") or url,
        "title": result.get("title", ""),
        "description": str(metadata.get("description") or ""),
        "content": result.get("content", ""),
    }


def _validation_message(error: ValidationError) -> str:
    """把 pydantic 校验错误转成调用方可直接修正的中文提示。

    调用方（其它 agent / MCP 客户端）只看到 tool 返回的错误文本，故不能
    直接把 pydantic 的英文原文抛出去：长度类错误需说明上限与改写建议。
    """
    first = error.errors()[0] if error.errors() else {}
    field = ".".join(str(part) for part in (first.get("loc") or [])) or "参数"
    etype = first.get("type")
    ctx = first.get("ctx") or {}
    value = first.get("input")
    if etype == "string_too_long":
        limit = ctx.get("max_length")
        actual = len(value) if isinstance(value, str) else None
        actual_text = f"（当前 {actual} 字符）" if actual else ""
        if field == "query":
            return (
                f"query 过长：最多 {limit} 字符{actual_text}；"
                "请改用简短关键词（如“上海 螺纹钢 信息价”），"
                "需要多个角度时拆成多次 web_search 调用"
            )
        return f"{field} 过长：最多 {limit} 字符{actual_text}"
    if etype in ("greater_than_equal", "less_than_equal"):
        bounds = ctx.get("ge") if etype == "greater_than_equal" else ctx.get("le")
        operator = "不小于" if etype == "greater_than_equal" else "不大于"
        return f"{field} 超出范围：{operator} {bounds}"
    if etype == "string_too_short":
        return f"{field} 不能为空"
    # pydantic 会给自定义 ValueError 加 "Value error, " 前缀，去掉后再返回。
    message = str(first.get("msg") or f"{field} 参数无效")
    return message.removeprefix("Value error, ")


def register_web_search_tools(mcp: FastMCP, *, config: ZhipuWebConfig) -> None:
    """把网络搜索 / 网页阅读 tool 注册到给定 MCP 服务（MCP 面）。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        config: 智谱 Coding Plan MCP 配置（api_key 必须已配置）。
    """

    @mcp.tool
    async def web_search(
        query: Annotated[
            str,
            PDField(
                description=(
                    "搜索关键词，最多 70 字符（上游硬限制）。"
                    "用简短关键词组合（如“上海 螺纹钢 信息价 2026”），"
                    "不要传整段长句或完整问题；需要多个角度时拆成多次调用。"
                )
            ),
        ],
        count: Annotated[
            int,
            PDField(
                description=(
                    "返回条数上限，1-50，默认 10。上游单次约返回 10 条，超出部分本地截断。"
                )
            ),
        ] = 10,
        recency: Annotated[
            str,
            PDField(
                description="时间范围：oneDay / oneWeek / oneMonth / oneYear / noLimit（默认）"
            ),
        ] = "noLimit",
        domain: Annotated[
            str | None,
            PDField(description="限定白名单域名，如 www.example.com；缺省不限"),
        ] = None,
        content_size: Annotated[
            str | None,
            PDField(description="内容量：medium（默认，摘要）/ high（更全面）"),
        ] = None,
        location: Annotated[
            str | None,
            PDField(description="地域：cn（默认中文区域）/ us（非中文区域）"),
        ] = None,
    ) -> dict[str, Any]:
        """网络搜索：返回与 query 相关的网页列表（标题、链接、摘要、引用编号）。

        注意 query 为上游硬限制的短关键词，最多 70 字符：长句会被拒绝，
        请先提炼关键词。

        Args:
            query: 搜索关键词，最多 70 字符，建议 3-8 个词的组合。
            count: 返回条数上限（1-50，默认 10）。
            recency: 时间范围（oneDay/oneWeek/oneMonth/oneYear/noLimit）。
            domain: 限定白名单域名（如 www.example.com）。
            content_size: 内容量（medium 摘要 / high 详细）。
            location: 地域（cn 中文 / us 非中文）。

        返回值:
            含 query、count、results 的字典；results 每项含 title、link、
            content、refer。
        """
        try:
            params = WebSearchInput(
                query=query,
                count=count,
                recency=recency,  # type: ignore[arg-type]
                domain=domain,
                content_size=content_size,  # type: ignore[arg-type]
                location=location,  # type: ignore[arg-type]
            )
        except ValidationError as error:
            raise ToolError(_validation_message(error)) from error
        try:
            return await _search(
                config,
                params.query,
                params.count,
                params.recency,
                params.domain,
                params.content_size,
                params.location,
            )
        except AppError as error:
            raise ToolError(error.public_message) from error

    @mcp.tool
    async def web_fetch(
        url: Annotated[
            str,
            PDField(description="要阅读的网页地址，必须是 http(s) 链接"),
        ],
        return_format: Annotated[
            str | None,
            PDField(description="返回格式：markdown（默认）/ text"),
        ] = None,
    ) -> dict[str, Any]:
        """网页阅读：抓取并解析指定 URL 的正文，返回 markdown（默认）或纯文本。

        Args:
            url: 必须是 http(s) 链接。
            return_format: markdown（默认）或 text。

        返回值:
            含 url、title、description、content 的字典；content 为解析后的正文。
        """
        try:
            params = WebFetchInput(url=url, return_format=return_format)  # type: ignore[arg-type]
        except ValidationError as error:
            raise ToolError(_validation_message(error)) from error
        try:
            return await _fetch(config, params.url, params.return_format)
        except AppError as error:
            raise ToolError(error.public_message) from error


class _WebSearchArgs(BaseModel):
    """Agent 面 web_search 的参数模型（描述供模型阅读，校验仍在 WebSearchInput）。"""

    query: str = PDField(
        min_length=1,
        description=(
            "搜索关键词，最多 70 字符（上游硬限制）：用简短关键词组合，"
            "不要传长句；需要多个角度时拆成多次调用"
        ),
    )
    count: int = PDField(default=10, description="返回条数上限 1-50，默认 10")
    recency: str = PDField(
        default="noLimit",
        description="时间范围：oneDay/oneWeek/oneMonth/oneYear/noLimit",
    )
    domain: str | None = PDField(default=None, description="限定白名单域名，如 www.example.com")
    content_size: str | None = PDField(
        default=None, description="内容量：medium（摘要）/ high（详细）"
    )
    location: str | None = PDField(default=None, description="地域：cn（默认）/ us")


class _WebFetchArgs(BaseModel):
    """Agent 面 web_fetch 的参数模型。"""

    url: str = PDField(min_length=1, description="要阅读的网页地址，必须是 http(s) 链接")
    return_format: str | None = PDField(
        default=None, description="返回格式：markdown（默认）/ text"
    )


def create_web_agent_tools(config: ZhipuWebConfig) -> list[BaseTool]:
    """构造 Deep Agent 自用的搜索/阅读工具（Agent 面，错误以字典返回）。

    参数:
        config: 智谱 Coding Plan MCP 配置（api_key 必须已配置）。

    返回值:
        含 web_search 与 web_fetch 两个工具的列表。
    """

    async def web_search(query: str, count: int = 10, **kwargs: Any) -> dict[str, Any]:
        """网络搜索：给定关键词返回网页列表；已知具体 URL 时改用 web_fetch。"""
        try:
            params = WebSearchInput(query=query, count=count, **kwargs)
        except ValidationError as error:
            return {"error": {"code": "validation", "message": _validation_message(error)}}
        try:
            return await _search(
                config,
                params.query,
                params.count,
                params.recency,
                params.domain,
                params.content_size,
                params.location,
            )
        except AppError as error:
            return {"error": {"code": error.code.value, "message": error.public_message}}

    async def web_fetch(url: str, return_format: str | None = None) -> dict[str, Any]:
        """网页阅读：抓取并解析指定 URL 的正文（markdown）。"""
        try:
            params = WebFetchInput(url=url, return_format=return_format)
        except ValidationError as error:
            return {"error": {"code": "validation", "message": _validation_message(error)}}
        try:
            return await _fetch(config, params.url, params.return_format)
        except AppError as error:
            return {"error": {"code": error.code.value, "message": error.public_message}}

    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description=(
                "网络搜索：给定关键词返回网页列表（标题/链接/摘要/引用编号）。"
                "已知具体 URL 时改用 web_fetch。query ≤70 字符，count 为条数上限（1-50）。"
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
