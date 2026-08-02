"""用 PyMuPDF 把 PDF 页面渲染为图片文件的节点。"""

import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pymupdf

from agent_app.errors import AppError, ErrorCode
from agent_app.workflows.pdf_to_image.pages import resolve_pages
from agent_app.workflows.pdf_to_image.schemas import PdfState

# 下载远程 PDF 的最大等待秒数。
PDF_DOWNLOAD_TIMEOUT = 30.0

# 对外暴露渲染图片的静态文件 URL 前缀，需与 main.py 中的挂载点保持一致。
STATIC_URL_PREFIX = "/static/pdf_images"


def _load_pdf_bytes(source: str | None, pdf_base64: str | None) -> bytes:
    """根据来源加载 PDF 原始字节；失败时抛出安全的 INVALID_PARAMETERS。

    参数:
        source: 可下载的 PDF URL，可为 None。
        pdf_base64: base64 编码的 PDF 字节，可为 None。

    返回值:
        可交给 PyMuPDF 打开的 PDF 原始字节。

    异常:
        AppError: 解码失败或下载失败时抛出 INVALID_PARAMETERS。
    """
    if pdf_base64 and pdf_base64.strip():
        try:
            return base64.b64decode(pdf_base64)
        except (binascii.Error, ValueError) as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                "PDF source could not be loaded",
            ) from error

    if source and source.strip():
        try:
            response = httpx.get(source.strip(), timeout=PDF_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                "PDF source could not be loaded",
            ) from error
        return response.content

    raise AppError(ErrorCode.INVALID_PARAMETERS, "PDF source could not be loaded")


def make_render_node(*, output_dir: Path) -> Callable[[PdfState], dict[str, Any]]:
    """返回把 PDF 渲染为图片文件并返回静态 URL 列表的同步节点。

    参数:
        output_dir: 渲染图片的落盘根目录；每次渲染在其中生成唯一子目录。

    返回值:
        接收 PDF 状态并返回公开结果字典的同步节点。
    """

    def render(state: PdfState) -> dict[str, Any]:
        """加载 PDF、渲染目标页并返回图片的静态 URL 与元数据。

        参数:
            state: 已校验的 PDF 转图片工作流状态。

        返回值:
            包含 images、page_count 和 rendered_pages 的结果字典。
        """
        raw = _load_pdf_bytes(state.get("source"), state.get("pdf_base64"))

        try:
            document = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as error:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                "PDF source could not be loaded",
            ) from error

        try:
            page_count = document.page_count
            indices = resolve_pages(
                state.get("pages"),
                state.get("page_ranges"),
                page_count,
            )
            dpi = state.get("dpi", 150)
            image_format = state.get("image_format", "png")
            matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
            render_id = uuid4().hex
            out_root = output_dir / render_id
            out_root.mkdir(parents=True, exist_ok=True)

            images: list[dict[str, Any]] = []
            for index in indices:
                pixmap = document[index].get_pixmap(matrix=matrix)
                file_name = f"page_{index + 1}.{image_format}"
                pixmap.save(str(out_root / file_name))
                images.append(
                    {
                        "page": index + 1,
                        "url": f"{STATIC_URL_PREFIX}/{render_id}/{file_name}",
                        "width": pixmap.width,
                        "height": pixmap.height,
                    }
                )
        except AppError:
            raise
        except Exception as error:
            raise AppError(ErrorCode.EXECUTION_FAILED, "PDF rendering failed") from error
        finally:
            document.close()

        return {
            "result": {
                "images": images,
                "page_count": page_count,
                "rendered_pages": [image["page"] for image in images],
            }
        }

    return render
