"""PDF 转图片 tool 的进程内实现与 MCP 注册。"""

from typing import Any, Literal
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from agent_app.errors import AppError
from agent_app.infrastructure.storage import ObjectStorage
from agent_app.tools.pdf.io import load_pdf_bytes
from agent_app.tools.pdf.render import render_pdf_pages
from agent_app.tools.pdf.schemas import PdfInput


def render_pdf_to_image(
    *,
    storage: ObjectStorage,
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    page_ranges: list[list[int]] | None = None,
    dpi: int = 150,
    image_format: Literal["png", "jpeg"] = "png",
) -> dict[str, Any]:
    """PDF 转图片的进程内实现：渲染指定页并上传到对象存储，返回每页可下载 URL。

    File（multipart 字节）与 URL（由调用方先行下载为字节）两种来源共用此入口。失败时
    抛出 ``AppError``，由调用方决定如何对外暴露——workflow 让其原样冒泡以保持稳定错误码，
    MCP 包装层将其转为 ``ToolError``。

    参数:
        storage: 上传渲染产物并换取下载 URL 的对象存储。
        pdf_bytes: 已加载的 PDF 原始字节。
        pages: 1-based 离散页码列表。
        page_ranges: 1-based 闭区间 ``[start, end]`` 列表。
        dpi: 渲染分辨率。
        image_format: 输出图片格式。

    返回值:
        含 images、page_count、rendered_pages 的字典。

    异常:
        AppError: 渲染失败或页码越界抛出 INVALID_PARAMETERS/EXECUTION_FAILED；
            上传失败抛出 UPSTREAM_UNAVAILABLE。
    """
    rendered, page_count = render_pdf_pages(
        pdf_bytes,
        pages=pages,
        page_ranges=page_ranges,
        dpi=dpi,
        image_format=image_format,
    )
    content_type = "image/png" if image_format == "png" else "image/jpeg"
    render_id = uuid4().hex
    images: list[dict[str, Any]] = []
    for item in rendered:
        key = f"pdf_images/{render_id}/page_{item.page}.{image_format}"
        storage.put(item.image_bytes, key=key, content_type=content_type)
        images.append(
            {
                "page": item.page,
                "url": storage.download_url(key),
                "width": item.width,
                "height": item.height,
            }
        )
    return {
        "images": images,
        "page_count": page_count,
        "rendered_pages": [image["page"] for image in images],
    }


def register_pdf_tools(
    mcp: FastMCP,
    *,
    storage: ObjectStorage,
) -> None:
    """把 PDF 转图片 tool 注册到给定 MCP 服务。

    参数:
        mcp: 待注册的 FastMCP 服务实例。
        storage: 上传渲染产物并换取下载 URL 的对象存储。
    """

    @mcp.tool
    def pdf_to_image(
        url: str,
        pages: list[int] | None = None,
        page_ranges: list[list[int]] | None = None,
        dpi: int = 150,
        image_format: Literal["png", "jpeg"] = "png",
    ) -> dict[str, Any]:
        """把 PDF 渲染为图片并上传，返回每页的可下载 URL。

        ``url`` 为可下载的 PDF URL（唯一来源）。页码可通过 ``pages``（离散）与
        ``page_ranges``（闭区间）同时指定，取并集；两者皆空默认渲染第一页。

        Args:
            url: 可下载的 PDF URL。
            pages: 1-based 离散页码列表。
            page_ranges: 1-based 闭区间 ``[start, end]`` 列表。
            dpi: 渲染分辨率，范围 72-600，默认 150。
            image_format: 输出图片格式 png 或 jpeg，默认 png。

        返回值:
            含 images、page_count、rendered_pages 的字典；image 项包含 page、url、width、height。
        """
        try:
            # 复用 PdfInput 做 URL/DPI/页码校验，保证与 workflow 行为一致。
            PdfInput(
                url=url,
                pages=pages,
                page_ranges=page_ranges,
                dpi=dpi,
                image_format=image_format,
            )
            return render_pdf_to_image(
                storage=storage,
                pdf_bytes=load_pdf_bytes(url),
                pages=pages,
                page_ranges=page_ranges,
                dpi=dpi,
                image_format=image_format,
            )
        except AppError as error:
            # 把安全的应用异常转成 MCP ToolError，public_message 可安全暴露给 client。
            raise ToolError(error.public_message) from error
        except ValidationError as error:
            detail = error.errors()[0]["msg"] if error.errors() else "Invalid PDF parameters"
            raise ToolError(str(detail)) from error
