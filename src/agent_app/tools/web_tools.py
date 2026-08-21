"""Web 网关能力的双面工具：MCP 注册与 Deep Agent langchain 工具。

两种暴露面共享同一 WebGatewayClient 与输入契约，但错误处理策略不同：
- MCP 面（register_web_tools）：AppError 转为 ToolError 抛出，由 FastMCP 边界
  安全转换为错误响应；
- Agent 面（create_web_agent_tools）：AppError 转为 {"error": {...}} 字典返回。
  这是有意为之——langchain ToolNode 默认仅把参数校验失败转为模型可见的错误
  消息，其余异常会原样穿出图边界导致整个任务 EXECUTION_FAILED；返回错误字典
  才能让模型在网关偶发故障时换查询、调小 max_chars 或放弃该 URL 继续任务。

Agent 面还实施全任务级调用预算（见 web_budget.py）：web_search/web_fetch
为 async 工具，执行前在事件循环线程内扣减 contextvar 预算，超限返回
错误字典促使模型收敛；同步网关调用经线程池下发，不阻塞事件循环。
"""

import functools
from typing import Any

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from agent_app.errors import AppError
from agent_app.infrastructure.web_gateway import WebGatewayClient
from agent_app.schemas.web_tools import WebFetchInput, WebSearchInput
from agent_app.tools.web_budget import budget_exceeded_payload, consume_web_call


def register_web_tools(
    mcp: FastMCP,
    *,
    client: WebGatewayClient,
) -> None:
    """把 Web 搜索/抓取 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        client: 执行搜索与抓取的 Web 网关客户端。
    """

    @mcp.tool
    def web_search(query: str, limit: int = 5) -> dict[str, Any]:
        """通过外部 Web 网关搜索互联网信息，返回结果列表。

        query 为搜索关键词；limit 可选，限制返回条数（1-10，默认 5）。

        Args:
            query: 搜索关键词。
            limit: 返回结果条数上限。

        返回值:
            网关原始响应字典。
        """
        try:
            payload = WebSearchInput(query=query, limit=limit)
            return client.search(query=payload.query, limit=payload.limit)
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid web search parameters"
            raise ToolError(str(detail)) from error

    @mcp.tool
    def web_fetch(url: str, max_chars: int = 20000) -> dict[str, Any]:
        """通过外部 Web 网关抓取网页正文。

        url 为目标网页地址；max_chars 可选，限制返回正文字符数（默认 20000）。

        Args:
            url: 目标网页地址。
            max_chars: 返回正文的字符数上限。

        返回值:
            网关原始响应字典。
        """
        try:
            payload = WebFetchInput(url=url, max_chars=max_chars)
            return client.fetch(url=payload.url, max_chars=payload.max_chars)
        except AppError as error:
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid web fetch parameters"
            raise ToolError(str(detail)) from error


def _as_agent_error(error: AppError) -> dict[str, Any]:
    """把 AppError 转为返回给模型的错误字典（不抛异常，见模块 docstring）。"""
    return {"error": {"code": error.code.value, "message": error.public_message}}


def create_web_agent_tools(client: WebGatewayClient) -> list[BaseTool]:
    """构造注入 Deep Agent 的 Web 搜索/抓取 langchain 工具。

    工具为 async 形式：每次执行前在事件循环线程内扣减全任务预算
    （contextvar，由 TaskService 初始化；未初始化时不限制），超限
    返回错误字典促使模型收敛。

    参数:
        client: 执行搜索与抓取的 Web 网关客户端。

    返回值:
        含 web_search 与 web_fetch 两个工具的列表。
    """

    @tool(args_schema=WebSearchInput)
    async def web_search(query: str, limit: int = 5) -> dict[str, Any]:
        """搜索互联网信息。

        需要了解时事、查资料或验证事实但不知道具体页面地址时使用；
        已知具体 URL 时改用 web_fetch。
        """
        if not consume_web_call():
            return budget_exceeded_payload()
        try:
            return await anyio.to_thread.run_sync(
                functools.partial(client.search, query=query, limit=limit)
            )
        except AppError as error:
            return _as_agent_error(error)

    @tool(args_schema=WebFetchInput)
    async def web_fetch(url: str, max_chars: int = 20000) -> dict[str, Any]:
        """抓取指定 URL 的网页正文。

        已知具体页面地址、需要阅读其完整内容时使用；
        只有关键词没有 URL 时改用 web_search。
        """
        if not consume_web_call():
            return budget_exceeded_payload()
        try:
            return await anyio.to_thread.run_sync(
                functools.partial(client.fetch, url=url, max_chars=max_chars)
            )
        except AppError as error:
            return _as_agent_error(error)

    return [web_search, web_fetch]
