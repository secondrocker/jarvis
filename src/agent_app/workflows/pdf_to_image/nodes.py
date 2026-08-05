"""把 PDF 渲染为图片的 LangGraph 节点（复用 tools 的 PDF tool 进程内实现）。"""

from collections.abc import Callable
from typing import Any

from agent_app.infrastructure.storage import ObjectStorage
from agent_app.tools.pdf.io import load_pdf_bytes
from agent_app.tools.pdf.tool import render_pdf_to_image
from agent_app.workflows.pdf_to_image.schemas import PdfState


def make_render_node(*, storage: ObjectStorage) -> Callable[[PdfState], dict[str, Any]]:
    """返回把 PDF 渲染为图片并上传对象存储、返回下载 URL 列表的同步节点。

    参数:
        storage: 上传渲染产物并换取下载 URL 的对象存储。

    返回值:
        接收 PDF 状态并返回公开结果字典的同步节点。
    """

    def render(state: PdfState) -> dict[str, Any]:
        """加载 PDF、渲染目标页并返回图片的下载 URL 与元数据。

        参数:
            state: 已校验的 PDF 转图片工作流状态。

        返回值:
            包含 images、page_count 和 rendered_pages 的结果字典。
        """
        result = render_pdf_to_image(
            storage=storage,
            pdf_bytes=load_pdf_bytes(state.get("url")),
            pages=state.get("pages"),
            page_ranges=state.get("page_ranges"),
            dpi=state.get("dpi", 150),
            image_format=state.get("image_format", "png"),
        )
        return {"result": result}

    return render
