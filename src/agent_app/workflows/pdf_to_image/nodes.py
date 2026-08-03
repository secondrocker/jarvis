"""把 PDF 渲染为图片文件的 LangGraph 节点（复用 tools 的 PDF tool 进程内实现）。"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_app.tools.pdf.persist import STATIC_URL_PREFIX
from agent_app.tools.pdf.tool import render_pdf_to_image
from agent_app.workflows.pdf_to_image.schemas import PdfState


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
        result = render_pdf_to_image(
            output_dir=output_dir,
            source=state.get("source"),
            pages=state.get("pages"),
            page_ranges=state.get("page_ranges"),
            dpi=state.get("dpi", 150),
            image_format=state.get("image_format", "png"),
            return_mode="url",
            url_prefix=STATIC_URL_PREFIX,
        )
        return {"result": result}

    return render
