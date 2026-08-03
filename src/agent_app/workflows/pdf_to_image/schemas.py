"""PDF 转图片工作流的数据模型与状态。

PdfInput 已迁移到 ``agent_app.tools.pdf.schemas``，此处保留别名以兼容旧 import。
"""

from typing import Any, TypedDict

from agent_app.tools.pdf.schemas import PdfInput

__all__ = ["PdfInput", "PdfState"]


class PdfState(TypedDict, total=False):
    """在 PDF 转图片图中流转的状态。"""

    source: str | None
    pages: list[int] | None
    page_ranges: list[list[int]] | None
    dpi: int
    image_format: str
    result: dict[str, Any]
