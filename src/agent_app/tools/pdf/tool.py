"""PDF 转图片 tool 的进程内实现与 MCP 注册。"""

import base64
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from agent_app.errors import AppError
from agent_app.tools.pdf.io import load_pdf_bytes
from agent_app.tools.pdf.persist import STATIC_URL_PREFIX, persist_rendered_images
from agent_app.tools.pdf.render import render_pdf_pages
from agent_app.tools.pdf.schemas import PdfInput


def render_pdf_to_image(
    *,
    output_dir: Path,
    source: str,
    pages: list[int] | None = None,
    page_ranges: list[list[int]] | None = None,
    dpi: int = 150,
    image_format: Literal["png", "jpeg"] = "png",
    return_mode: Literal["base64", "url"] = "url",
    url_prefix: str = STATIC_URL_PREFIX,
) -> dict[str, Any]:
    """PDF 转图片的进程内实现：下载、渲染、持久化或 base64 编码。

    workflow 节点与 MCP tool 共用此入口。失败时抛出 ``AppError``，由调用方决定如何对外
    暴露——workflow 让其原样冒泡以保持稳定错误码，MCP 包装层将其转为 ``ToolError``。

    参数:
        output_dir: url 模式的落盘根目录（base64 模式不使用）。
        source: 可下载的 PDF URL。
        pages: 1-based 离散页码列表。
        page_ranges: 1-based 闭区间 ``[start, end]`` 列表。
        dpi: 渲染分辨率。
        image_format: 输出图片格式。
        return_mode: ``url``（默认，写盘 + 静态 URL）或 ``base64``（自包含字节）。
        url_prefix: url 模式的静态 URL 前缀。

    返回值:
        含 images、page_count、rendered_pages 的字典。

    异常:
        AppError: 下载或渲染失败时抛出 INVALID_PARAMETERS 或 EXECUTION_FAILED。
    """
    raw = load_pdf_bytes(source)
    rendered, page_count = render_pdf_pages(
        raw,
        pages=pages,
        page_ranges=page_ranges,
        dpi=dpi,
        image_format=image_format,
    )
    if return_mode == "url":
        images = persist_rendered_images(
            rendered,
            output_dir=output_dir,
            url_prefix=url_prefix,
            image_format=image_format,
        )
    else:
        images = [
            {
                "page": item.page,
                "data": base64.b64encode(item.image_bytes).decode("ascii"),
                "width": item.width,
                "height": item.height,
            }
            for item in rendered
        ]
    return {
        "images": images,
        "page_count": page_count,
        "rendered_pages": [image["page"] for image in images],
    }


def register_pdf_tools(
    mcp: FastMCP,
    *,
    output_dir: Path,
    url_prefix: str,
) -> None:
    """把 PDF 转图片 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        output_dir: url 模式的落盘根目录。
        url_prefix: url 模式的静态 URL 前缀。
    """

    @mcp.tool
    def pdf_to_image(
        source: str,
        pages: list[int] | None = None,
        page_ranges: list[list[int]] | None = None,
        dpi: int = 150,
        image_format: Literal["png", "jpeg"] = "png",
        return_mode: Literal["base64", "url"] = "base64",
    ) -> dict[str, Any]:
        """把 PDF 渲染为图片，返回每页的 base64 编码或静态 URL。

        ``source`` 为可下载的 PDF URL（唯一来源）。
        页码可通过 ``pages``（离散）与 ``page_ranges``（闭区间）同时指定，取并集；
        两者皆空默认渲染第一页。大文件或多页建议用 ``return_mode="url"`` 以避免 base64 体积过大。

        Args:
            source: 可下载的 PDF URL。
            pages: 1-based 离散页码列表。
            page_ranges: 1-based 闭区间 ``[start, end]`` 列表。
            dpi: 渲染分辨率，范围 72-600，默认 150。
            image_format: 输出图片格式 png 或 jpeg，默认 png。
            return_mode: ``base64``（默认，自包含、跨进程通用）或
                ``url``（需服务端同进程静态服务）。

        返回值:
            含 images、page_count、rendered_pages 的字典；image 项包含 page、width、height，
            以及 base64 模式下的 ``data`` 或 url 模式下的 ``url``。
        """
        try:
            # 复用 PdfInput 做 DPI/页码/来源校验，保证与 workflow 行为一致。
            PdfInput(
                source=source,
                pages=pages,
                page_ranges=page_ranges,
                dpi=dpi,
                image_format=image_format,
            )
            return render_pdf_to_image(
                output_dir=output_dir,
                url_prefix=url_prefix,
                source=source,
                pages=pages,
                page_ranges=page_ranges,
                dpi=dpi,
                image_format=image_format,
                return_mode=return_mode,
            )
        except AppError as error:
            # 把安全的应用异常转成 MCP ToolError，public_message 可安全暴露给 client。
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid PDF parameters"
            raise ToolError(str(detail)) from error
