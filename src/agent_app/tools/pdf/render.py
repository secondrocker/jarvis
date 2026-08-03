"""PDF 页面渲染为图片字节的纯核心（无 I/O、无 LLM、无副作用）。"""

from dataclasses import dataclass
from typing import Literal

import pymupdf

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.pdf.pages import resolve_pages


@dataclass(frozen=True)
class RenderedPage:
    """单页渲染产物（自包含字节，调用方自行决定落盘或 base64）。

    属性:
        page: 1-based 页码。
        image_bytes: 指定格式编码的图片字节（PNG/JPEG）。
        width: 渲染后的像素宽度。
        height: 渲染后的像素高度。
    """

    page: int
    image_bytes: bytes
    width: int
    height: int


def render_pdf_pages(
    pdf_bytes: bytes,
    *,
    pages: list[int] | None = None,
    page_ranges: list[list[int]] | None = None,
    dpi: int = 150,
    image_format: Literal["png", "jpeg"] = "png",
) -> tuple[list[RenderedPage], int]:
    """把 PDF 指定页渲染为图片字节，返回渲染结果与 PDF 总页数。

    纯函数：仅依赖输入字节，不触网络、不落盘。DPI 仅用于缩放矩阵。

    参数:
        pdf_bytes: PDF 原始字节。
        pages: 1-based 离散页码列表，可为 None。
        page_ranges: 1-based 闭区间 ``[start, end]`` 列表，可为 None。
        dpi: 渲染分辨率，相对 72 DPI 的缩放系数。
        image_format: 输出图片格式。

    返回值:
        由 (渲染结果列表, PDF 总页数) 组成的元组。

    异常:
        AppError: 打开失败或页码越界抛出 INVALID_PARAMETERS；渲染异常抛出 EXECUTION_FAILED。
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise AppError(
            ErrorCode.INVALID_PARAMETERS,
            "PDF source could not be loaded",
        ) from error

    try:
        page_count = document.page_count
        indices = resolve_pages(pages, page_ranges, page_count)
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        rendered: list[RenderedPage] = []
        for index in indices:
            pixmap = document[index].get_pixmap(matrix=matrix)
            rendered.append(
                RenderedPage(
                    page=index + 1,
                    image_bytes=pixmap.tobytes(output=image_format),
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
    except AppError:
        raise
    except Exception as error:
        raise AppError(ErrorCode.EXECUTION_FAILED, "PDF rendering failed") from error
    finally:
        document.close()

    return rendered, page_count
